"""One learned yaw about the fixed tool/platform normal; no spatial pose field."""
import json
import math
from pathlib import Path

import torch

from globalOrientation import parse_schedule, scheduled_value


def axial_rotation(angle, axis):
    axis = axis / axis.norm().clamp_min(1e-12)
    x, y, z = axis.unbind()
    zero = x*0
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return eye + torch.sin(angle)*skew + (1-torch.cos(angle))*(skew@skew)


class AxialOrientation:
    """The same training hook as GlobalOrientation, with one scalar parameter."""
    def __init__(self, args, output):
        self.args, self.output = args, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.update_epochs=tuple(sorted(getattr(args,'yaw_update_epochs',None) or ()))
        self.update_steps=int(getattr(args,'yaw_update_steps',10))
        self.schedule = [(1,'field')] if self.update_epochs else parse_schedule(args.orientation_schedule)
        self.metadata = dict(enabled=True, mode='fixed_axis_yaw', schedule=self.schedule,
            learning_rate=args.orientation_learning_rate,
            parameterization='one global angle in radians about the fixed tool axis',
            initial_delta_degrees=0., convention='positive right-hand rotation of tool/platform frame; inverse part yaw on export',
            base_policy='normal and existing base constraints unchanged; finite bed fit checked at machine export')
        self.metadata.update(update_epochs=list(self.update_epochs),steps_per_update=self.update_steps,
            update_policy='intermediate yaw-only bursts' if self.update_epochs else 'scheduled joint/field/frame phases')
        self.steps = 0
        self.last_mining_step = -args.orientation_refresh_steps

    @property
    def parameter(self):
        return self.angle

    def bind(self, field, frame, scale):
        parameter = next(field.parameters())
        self.angle = torch.nn.Parameter(parameter.new_zeros(()))
        self.frame = tuple(vector.to(parameter) for vector in frame)
        if not torch.allclose(scale, scale[0].expand_as(scale)):
            raise ValueError('Global yaw requires isotropic model normalization.')
        self.original_requires_grad = [p.requires_grad for p in field.parameters()]
        self.metadata['reference_frame_model'] = dict(zip(('u', 'v', 'n'), [v.detach().cpu().tolist() for v in self.frame]))
        self.initial_report = self.report()
        (self.output/'initial_orientation.json').write_text(json.dumps(self.initial_report, indent=2))

    def optimizer(self, field, learning_rate):
        return torch.optim.Adam([dict(params=field.parameters(), lr=learning_rate, name='field'),
            dict(params=[self.angle], lr=self.args.orientation_learning_rate, name='orientation')])

    def begin_epoch(self, epoch, field, optimizer):
        self.epoch = epoch+1
        self.phase = scheduled_value(self.schedule, self.epoch)
        for parameter, original in zip(field.parameters(), self.original_requires_grad):
            parameter.requires_grad_(original and self.phase != 'frame')
        self.angle.requires_grad_(self.phase != 'field')
        optimizer.param_groups[1]['lr'] = self.args.orientation_learning_rate
        self.gradient_sum = self.angle.new_zeros(())
        self.epoch_steps = 0
        self.last_update = None

    def matrix(self):
        return axial_rotation(self.angle, self.frame[2])

    def rotate_offsets(self, points):
        return points @ self.matrix().T

    def direction(self):
        return self.frame[2]  # Invariant: yaw cannot tilt the tool/platform normal.

    @torch.no_grad()
    def project(self):
        self.angle.copy_(torch.remainder(self.angle+math.pi, 2*math.pi)-math.pi)

    def after_step(self):
        if not self.angle.requires_grad:
            return  # Frozen means bitwise unchanged, including projection and optimizer state.
        if self.angle.grad is not None:
            self.gradient_sum += self.angle.grad.detach().abs()
        self.project(); self.steps += 1; self.epoch_steps += 1

    def intermediate_update(self,module,field,optimizer,root_points,tool_points,values,counts,collision_weight):
        """Insert a short yaw-only burst without consuming a field epoch."""
        if self.epoch not in self.update_epochs:
            return
        if collision_weight<=0:
            raise ValueError('Intermediate yaw updates require positive collision weight.')
        parameters=list(field.parameters());flags=[p.requires_grad for p in parameters]
        angle_flag=self.angle.requires_grad;old_rate=optimizer.param_groups[1]['lr']
        before=float(self.angle.detach().cpu())
        try:
            for parameter in parameters:parameter.requires_grad_(False)
            self.angle.requires_grad_(True);optimizer.param_groups[1]['lr']=self.args.orientation_learning_rate
            for _ in range(self.update_steps):
                optimizer.zero_grad(set_to_none=True)
                out=module.compute_gated_translated_scalar_loss(scalar_field=field,
                    root_points=root_points.detach(),tool_points=tool_points,
                    gating_field=values.get('sdf_gate_model'),gate_epsilon=values['gate_epsilon'],
                    x_min=values.get('x_min'),x_max=values.get('x_max'),y_min=values.get('y_min'),
                    y_max=values.get('y_max'),z_min=values.get('z_min'),z_max=values.get('z_max'))
                groups=counts or [len(tool_points)]
                loss,_=module.reduce_collision_loss(out,groups,module.COLLISION_LOSS_REDUCTION,
                    module.COLLISION_TOPK_FRACTION,module.COLLISION_TOOL_WEIGHTS[:len(groups)])
                (collision_weight*loss+self.angle*0).backward();optimizer.step();self.after_step()
            self.last_update=dict(after_field_epoch=self.epoch,steps=self.update_steps,
                yaw_delta_before_degrees=math.degrees(before),yaw_delta_after_degrees=math.degrees(float(self.angle.detach().cpu())))
            manager=getattr(module,'_hard_collision_manager',None)
            if manager is not None:manager.refresh_pending=True
            print('intermediate yaw '+json.dumps(self.last_update),flush=True)
        finally:
            optimizer.zero_grad(set_to_none=True)
            for parameter,flag in zip(parameters,flags):parameter.requires_grad_(flag)
            self.angle.requires_grad_(angle_flag);optimizer.param_groups[1]['lr']=old_rate

    def report(self):
        with torch.no_grad():
            matrix = self.matrix()
            u, v, n = [(matrix@item).cpu().tolist() for item in self.frame]
            angle = float(self.angle.detach().cpu())
            quaternion = torch.cat((torch.cos(self.angle/2).reshape(1),
                torch.sin(self.angle/2)*self.frame[2])).cpu().tolist()
        return dict(yaw_delta_degrees=math.degrees(angle), quaternion=quaternion,
            rotation_degrees=abs(math.degrees(angle)), rotation_model_from_reference=matrix.cpu().tolist(),
            rotation_reference_from_model=matrix.T.cpu().tolist(), tool_frame_model=dict(u=u, v=v, n=n),
            metadata=self.metadata)

    def checkpoint(self):
        report = self.report()
        return dict(learned_quaternion=torch.tensor(report['quaternion']),
            learned_yaw_delta_degrees=report['yaw_delta_degrees'],
            learned_tool_frame=report['tool_frame_model'], orientation_search=self.metadata)

    def end_epoch(self, collision_weight):
        report = self.report()
        report.update(epoch=self.epoch, phase=self.phase, steps=self.epoch_steps,
            collision_weight=collision_weight,
            intermediate_update=self.last_update,
            mean_yaw_gradient_norm=float(self.gradient_sum/max(1, self.epoch_steps)))
        with (self.output/'orientation_metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(report)+'\n')
        (self.output/'orientation.json').write_text(json.dumps(report, indent=2))
        print('global yaw '+json.dumps({k:report[k] for k in ('epoch', 'phase', 'yaw_delta_degrees', 'mean_yaw_gradient_norm')}), flush=True)

    def finish(self, field):
        report = self.report()
        report['bed_feasibility'] = 'Normal and plane unchanged by yaw; finite plate bounds are checked by machine export.'
        (self.output/'orientation_result.json').write_text(json.dumps(report, indent=2))
        for parameter, original in zip(field.parameters(), self.original_requires_grad):
            parameter.requires_grad_(original)
