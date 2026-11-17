"""One bounded global tool rotation; scalar data remain in model coordinates."""
import json
import math
from pathlib import Path

import numpy as np
import torch


def parse_schedule(items, kind='phase'):
    allowed = {'field', 'frame', 'joint'}
    parsed = []
    for item in items:
        start, value = item.split(':', 1)
        start = int(start)
        value = value if kind == 'phase' else float(value)
        if start < 1 or (kind == 'phase' and value not in allowed):
            raise ValueError('Schedule entries need positive epochs and field/frame/joint phases')
        if kind != 'phase' and (not math.isfinite(value) or value < 0):
            raise ValueError('Collision schedule weights must be finite and nonnegative')
        parsed.append((start, value))
    parsed.sort()
    if not parsed or parsed[0][0] != 1 or len({p[0] for p in parsed}) != len(parsed):
        raise ValueError('Schedules must start at epoch 1 and have unique transition epochs')
    return parsed


def scheduled_value(schedule, epoch):
    return next(value for start, value in reversed(schedule) if start <= epoch)


def normalize_quaternion(q):
    return q / q.norm().clamp_min(1e-12)


def rotation_matrix(q):
    w, x, y, z = normalize_quaternion(q).unbind()
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    )).reshape(3, 3)


class GlobalOrientation:
    @property
    def parameter(self):
        return self.quaternion

    def __init__(self, args, output):
        self.args, self.output = args, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.schedule = parse_schedule(args.orientation_schedule)
        initial = args.orientation_initial_quaternion
        if initial is None and args.start_checkpoint is not None:
            checkpoint = torch.load(args.start_checkpoint, map_location='cpu', weights_only=True)
            initial = checkpoint.get('learned_quaternion')
        self.initial = torch.as_tensor([1., 0, 0, 0] if initial is None else initial).reshape(4)
        if not torch.isfinite(self.initial).all() or self.initial.norm() < 1e-10:
            raise ValueError('Initial quaternion must be finite and nonzero')
        self.initial = normalize_quaternion(self.initial)
        self.metadata = dict(enabled=True, schedule=self.schedule,
            learning_rate=args.orientation_learning_rate, max_rotation_degrees=args.orientation_max_rotation,
            initial_quaternion=self.initial.tolist(), convention='w,x,y,z; model-from-reference tool rotation',
            base_policy='retain original selected base constraints; bed feasibility is not optimized',
            parameterization='one global quaternion, not a spatially varying tool pose')
        self.steps = 0
        self.last_mining_step = -args.orientation_refresh_steps

    def bind(self, field, frame, scale):
        parameter = next(field.parameters())
        self.quaternion = torch.nn.Parameter(self.initial.to(parameter))
        self.frame = tuple(vector.to(parameter) for vector in frame)
        if not torch.allclose(scale, scale[0].expand_as(scale)):
            raise ValueError('Global tool rotation requires isotropic model normalization')
        self.original_requires_grad = [p.requires_grad for p in field.parameters()]
        self.project()
        self.initial_report = self.report()
        (self.output/'initial_orientation.json').write_text(json.dumps(self.initial_report, indent=2))

    def optimizer(self, field, learning_rate):
        return torch.optim.Adam([
            dict(params=field.parameters(), lr=learning_rate, name='field'),
            dict(params=[self.quaternion], lr=self.args.orientation_learning_rate, name='orientation'),
        ])

    def begin_epoch(self, epoch, field, optimizer):
        self.epoch = epoch+1
        self.phase = scheduled_value(self.schedule, self.epoch)
        for parameter, original in zip(field.parameters(), self.original_requires_grad):
            parameter.requires_grad_(original and self.phase != 'frame')
        self.quaternion.requires_grad_(self.phase != 'field')
        optimizer.param_groups[1]['lr'] = self.args.orientation_learning_rate
        self.gradient_sum = self.quaternion.new_zeros(())
        self.epoch_steps = 0

    def rotate_offsets(self, points):
        # Rebuild for each collision call; never reuse a rotation graph across steps.
        return points @ rotation_matrix(self.quaternion).T

    def direction(self):
        return rotation_matrix(self.quaternion) @ self.frame[2]

    @torch.no_grad()
    def project(self):
        q = normalize_quaternion(self.quaternion)
        q = torch.where(q[0] < 0, -q, q)
        half_angle = torch.atan2(q[1:].norm(), q[0].clamp_min(0))
        bound = q.new_tensor(math.radians(self.args.orientation_max_rotation)/2)
        angle = torch.minimum(half_angle, bound)
        vector = q[1:] / q[1:].norm().clamp_min(1e-12) * torch.sin(angle)
        self.quaternion.copy_(torch.cat((torch.cos(angle).reshape(1), vector)))

    def after_step(self):
        if self.quaternion.grad is not None:
            self.gradient_sum += self.quaternion.grad.detach().norm()
        self.project()
        self.steps += 1
        self.epoch_steps += 1

    def checkpoint(self):
        return dict(learned_quaternion=normalize_quaternion(self.quaternion.detach()).cpu(),
                    orientation_search=self.metadata,
                    learned_tool_frame=self.report()['tool_frame_model'])

    def report(self):
        with torch.no_grad():
            matrix = rotation_matrix(self.quaternion).cpu().numpy()
            u, v, n = [(rotation_matrix(self.quaternion) @ item).cpu().tolist() for item in self.frame]
            q = normalize_quaternion(self.quaternion).cpu()
        return dict(quaternion=q.tolist(), rotation_model_from_reference=matrix.tolist(),
                    rotation_reference_from_model=matrix.T.tolist(),
                    rotation_degrees=float(2*torch.atan2(q[1:].norm(), q[0].abs())*180/math.pi),
                    tool_frame_model=dict(u=u, v=v, n=n), metadata=self.metadata)

    def end_epoch(self, collision_weight):
        report = self.report()
        report.update(epoch=self.epoch, phase=self.phase, steps=self.epoch_steps,
                      collision_weight=collision_weight,
                      mean_quaternion_gradient_norm=float(self.gradient_sum/max(1, self.epoch_steps)))
        with (self.output/'orientation_metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(report)+'\n')
        (self.output/'orientation.json').write_text(json.dumps(report, indent=2))
        print('orientation '+json.dumps({k:report[k] for k in ['epoch','phase','rotation_degrees','mean_quaternion_gradient_norm']}))

    def finish(self, field):
        import igl
        report = self.report()
        vertices, faces = igl.read_triangle_mesh(str(self.args.mesh_path))
        center = (vertices.min(0)+vertices.max(0))/2
        matrix = np.asarray(report['rotation_model_from_reference'])
        n = self.frame[2].detach().cpu().numpy()
        n = n/np.linalg.norm(n)
        rotated = (vertices-center) @ matrix
        shift = -float((rotated @ n).min())*n
        rotated += shift
        transform = np.eye(4)
        transform[:3,:3] = matrix.T
        transform[:3,3] = shift-matrix.T@center
        report['model_to_machine_transform'] = transform.tolist()
        report['machine_up'] = n.tolist()
        report['bed_feasibility'] = 'not established: selected base constraints were retained in original model coordinates'
        original_projection = vertices@n
        base_band = original_projection <= original_projection.min()+self.args.base_height
        learned_n = np.asarray(report['tool_frame_model']['n'])
        report['original_base_band_depth_along_learned_axis_mm'] = float(np.ptp(vertices[base_band]@learned_n)) if base_band.any() else None
        (self.output/'orientation_result.json').write_text(json.dumps(report, indent=2))
        igl.write_triangle_mesh(str(self.output/'oriented_mesh.obj'), rotated, faces)
        for parameter, original in zip(field.parameters(), self.original_requires_grad):
            parameter.requires_grad_(original)
