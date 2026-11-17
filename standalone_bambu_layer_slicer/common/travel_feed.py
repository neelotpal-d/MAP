"""Limit the Z component of nonextruding body moves and preserve modal feeds."""
import math
import re

NUMBER=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)'
AXES=re.compile(r'(?<!\S)([XYZ])('+NUMBER+r')(?=\s|$)')
FEED=re.compile(r'(?<!\S)F('+NUMBER+r')(?=\s|$)')
EXTRUSION=re.compile(r'(?<!\S)E('+NUMBER+r')(?=\s|$)')
PRINT=re.compile(r'^G1\s+X('+NUMBER+r')\s+Y('+NUMBER+r')\s+Z('+NUMBER+r')\s+E('+NUMBER+r')\s+F('+NUMBER+r')\s*$')


class TravelFeedLimiter:
    """Transform one complete line; input and emitted modal F remain distinct."""
    def __init__(self, maximum_z_speed):
        self.limit=float(maximum_z_speed)
        if not math.isfinite(self.limit) or self.limit<=0:raise ValueError('Positive finite travel Z speed required.')
        self.position=[None]*3;self.source_feed=None;self.output_feed=None
        self.limited_moves=0;self.feed_restores=0

    def line(self,line):
        code=line.partition(';')[0].strip()
        if not code.startswith(('G0 ','G1 ')):return line
        # Most output is explicit depositing G1: preserve its bytes and feed.
        match=PRINT.fullmatch(code)
        if match and float(match[4])>0:
            self.position=[float(match[i]) for i in (1,2,3)]
            self.source_feed=self.output_feed=float(match[5]);return line
        old=self.position.copy()
        for axis,value in AXES.findall(code):self.position['XYZ'.index(axis)]=float(value)
        f=FEED.search(line.partition(';')[0])
        if f:self.source_feed=float(f[1])
        desired=self.source_feed;e=EXTRUSION.search(code);amount=float(e[1]) if e else 0.
        if desired is not None and amount<=0 and None not in old+self.position:
            length=math.dist(old,self.position);dz=abs(self.position[2]-old[2])
            if length>0 and dz>0:
                capped=min(desired,60*self.limit*length/dz)
                if capped<desired-1e-8:
                    desired=math.floor(capped*1e5)/1e5;self.limited_moves+=1
        if desired is None:return line
        if desired!=self.source_feed or (not f and self.output_feed is not None and abs(self.output_feed-desired)>1e-8):
            if desired==self.source_feed:self.feed_restores+=1
            token=f'F{desired:.5f}'.rstrip('0').rstrip('.')
            if f:line=line[:f.start()]+token+line[f.end():]
            else:
                text=line.rstrip('\r\n');command,sep,comment=text.partition(';')
                ending=line[len(line.rstrip('\r\n')):]
                line=command.rstrip()+' '+token+(' '+sep+comment if sep else '')+ending
        self.output_feed=desired
        return line


def limit_file(source,destination,maximum_z_speed=20.):
    limiter=TravelFeedLimiter(maximum_z_speed);body=False
    with open(source,encoding='utf8',newline='') as incoming,open(destination,'x',encoding='utf8',newline='') as out:
        for line in incoming:
            if line.strip()==';===== assembled layer path body start =====':body=True
            if line.strip()==';===== assembled layer path body end =====':body=False
            out.write(limiter.line(line) if body else line)
    return dict(limited_moves=limiter.limited_moves,modal_feed_restores=limiter.feed_restores)
