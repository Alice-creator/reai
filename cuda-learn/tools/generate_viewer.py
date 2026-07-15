"""
Generate roadmap.html with embedded lesson viewer.
Run: python generate_viewer.py
"""
import json, re
from pathlib import Path

LESSONS_DIR = Path('/home/loc-dev/Projects/neon/cuda-learn/lessons')
OUT_FILE    = Path('/home/loc-dev/Projects/neon/cuda-learn/roadmap.html')

# ── Load scraped lessons ─────────────────────────────────────────────────────
lessons = {}
for f in LESSONS_DIR.glob('*.json'):
    if f.name == '_all.json': continue
    d = json.loads(f.read_text())
    slug = d.get('slug', f.stem)
    lessons[slug] = {
        'title':        d.get('title', slug),
        'section':      d.get('section', ''),
        'topic':        d.get('topic', ''),
        'description':  d.get('description', '')[:4000],
        'url':          d.get('url', ''),
        'html_content': d.get('html_content', ''),
    }


# ── Canvas visualizations (JS bodies) ──────────────────────────────────────
VISUALIZATIONS = {

'calculate-mean': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',BLU='#60a5fa',GRN='#4ade80';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const data=[2,3,4,4,5,5,5,6,6,7,8,9];
  const mean=data.reduce((a,b)=>a+b,0)/data.length;
  const counts=new Array(11).fill(0);
  data.forEach(v=>counts[v]++);
  const maxC=Math.max(...counts);
  const pad={l:40,r:20,t:20,b:40};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  const barW=cw/10;
  for(let v=0;v<=10;v++){
    const c=counts[v]; if(c===0) continue;
    const bh=(c/maxC)*ch*0.85;
    const x=pad.l+v*(cw/10), y=pad.t+ch-bh;
    ctx.fillStyle='#1e40af44'; ctx.fillRect(x+2,y,barW-4,bh);
    ctx.strokeStyle='#3b82f6'; ctx.lineWidth=1; ctx.strokeRect(x+2,y,barW-4,bh);
  }
  ctx.fillStyle=TXT; ctx.font='10px monospace'; ctx.textAlign='center';
  for(let v=0;v<=10;v+=2){ const x=pad.l+v*(cw/10)+barW/2; ctx.fillText(v,x,pad.t+ch+14); }
  ctx.textAlign='right';
  for(let c=0;c<=maxC;c++){ const y=pad.t+ch-(c/maxC)*ch*0.85; ctx.fillText(c,pad.l-4,y+3); }
  const mx=pad.l+mean*(cw/10)+barW/2;
  ctx.strokeStyle=BLU; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(mx,pad.t); ctx.lineTo(mx,pad.t+ch); ctx.stroke();
  ctx.fillStyle=BLU; ctx.textAlign='center'; ctx.font='bold 10px monospace';
  ctx.fillText('mu='+mean.toFixed(1),mx,pad.t+10);
  const sorted=[...data].sort((a,b)=>a-b);
  const med=(sorted[5]+sorted[6])/2;
  const medx=pad.l+med*(cw/10)+barW/2;
  ctx.strokeStyle=GRN; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(medx,pad.t); ctx.lineTo(medx,pad.t+ch); ctx.stroke();
  ctx.setLineDash([]);
  ctx.font='10px monospace'; ctx.textAlign='left';
  ctx.fillStyle=BLU; ctx.fillRect(W-100,8,10,3); ctx.fillText('Mean',W-87,13);
  ctx.fillStyle=GRN; ctx.fillRect(W-100,20,10,3); ctx.fillText('Median',W-87,25);
""",

'calculate-variance-std': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',BLU='#60a5fa';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:44,r:20,t:24,b:36};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const xMin=-4,xMax=4;
  function gauss(x,mu,sig){return Math.exp(-0.5*((x-mu)/sig)**2)/(sig*Math.sqrt(2*Math.PI));}
  function toCanvasX(x){return pad.l+(x-xMin)/(xMax-xMin)*cw;}
  function toCanvasY(y,yMax){return pad.t+ch-(y/yMax)*ch*0.9;}
  const yMax=gauss(0,0,1)*1.1;
  ctx.fillStyle='rgba(96,165,250,0.08)';
  ctx.beginPath(); ctx.moveTo(toCanvasX(-2),pad.t+ch);
  for(let xi=-2;xi<=2;xi+=0.05){const y=gauss(xi,0,1);ctx.lineTo(toCanvasX(xi),toCanvasY(y,yMax));}
  ctx.lineTo(toCanvasX(2),pad.t+ch); ctx.closePath(); ctx.fill();
  ctx.fillStyle='rgba(96,165,250,0.18)';
  ctx.beginPath(); ctx.moveTo(toCanvasX(-1),pad.t+ch);
  for(let xi=-1;xi<=1;xi+=0.05){const y=gauss(xi,0,1);ctx.lineTo(toCanvasX(xi),toCanvasY(y,yMax));}
  ctx.lineTo(toCanvasX(1),pad.t+ch); ctx.closePath(); ctx.fill();
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  ctx.strokeStyle=BLU; ctx.lineWidth=2; ctx.beginPath();
  for(let xi=xMin;xi<=xMax;xi+=0.05){
    const cx2=toCanvasX(xi),cy=toCanvasY(gauss(xi,0,1),yMax);
    xi===xMin?ctx.moveTo(cx2,cy):ctx.lineTo(cx2,cy);
  }
  ctx.stroke();
  const sigs=[-2,-1,0,1,2]; const labels=['-2s','-s','u','+s','+2s'];
  sigs.forEach((s,i)=>{
    ctx.strokeStyle=i===2?'#ffffff44':'#33415544'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(toCanvasX(s),pad.t); ctx.lineTo(toCanvasX(s),pad.t+ch); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
    ctx.fillText(labels[i],toCanvasX(s),pad.t+ch+14);
  });
  ctx.fillStyle='rgba(96,165,250,0.9)'; ctx.font='10px monospace'; ctx.textAlign='center';
  ctx.fillText('+-1s (68%)',toCanvasX(0),toCanvasY(gauss(0.5,0,1),yMax)-6);
  ctx.fillStyle='rgba(96,165,250,0.5)';
  ctx.fillText('+-2s (95%)',toCanvasX(1.6),toCanvasY(gauss(1.6,0,1),yMax)-18);
""",

'monte-carlo-pi': """\
  const BG='#0a0a0f',AXES='#334155',GRN='#4ade80',RED='#f87171',BLU='#60a5fa';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad=30; const sz=Math.min(W,H)-pad*2;
  const ox=Math.floor((W-sz)/2), oy=Math.floor((H-sz)/2);
  ctx.strokeStyle=AXES; ctx.lineWidth=1; ctx.strokeRect(ox,oy,sz,sz);
  ctx.strokeStyle=BLU; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.arc(ox,oy+sz,sz,-(Math.PI/2),0); ctx.stroke();
  let inside=0;
  for(let i=0;i<200;i++){
    const px=Math.abs(Math.sin(i*2.399)*0.5+Math.cos(i*1.618)*0.5);
    const py=Math.abs(Math.cos(i*2.399)*0.5+Math.sin(i*1.618)*0.5);
    const isIn=(px*px+py*py)<1;
    if(isIn) inside++;
    const cx=ox+px*sz, cy=oy+(1-py)*sz;
    ctx.fillStyle=isIn?GRN:RED;
    ctx.beginPath(); ctx.arc(cx,cy,2,0,Math.PI*2); ctx.fill();
  }
  const piEst=4*inside/200;
  ctx.fillStyle='#e2e8f0'; ctx.font='bold 13px monospace'; ctx.textAlign='center';
  ctx.fillText('pi ~ '+piEst.toFixed(3),W/2,pad-6);
  ctx.fillStyle=AXES; ctx.font='9px monospace';
  ctx.fillText('Green: inside   Red: outside',W/2,H-8);
""",

'pearson-correlation': """\
  const BG='#0a0a0f',TXT='#64748b',GRN='#4ade80',ORG='#fb923c';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const panels=[{r:0.95,label:'r~+1',col:GRN},{r:0,label:'r~0',col:TXT},{r:-0.95,label:'r~-1',col:ORG}];
  const pw=Math.floor(W/3)-10; const ph=H-40;
  panels.forEach((panel,pi)=>{
    const ox=pi*(W/3)+5; const oy=20;
    ctx.strokeStyle='#1e293b'; ctx.lineWidth=1; ctx.strokeRect(ox,oy,pw,ph);
    ctx.fillStyle=panel.col; ctx.font='bold 10px monospace'; ctx.textAlign='center';
    ctx.fillText(panel.label,ox+pw/2,oy-6);
    const n=15; const pts=[];
    for(let i=0;i<n;i++){
      const x=(i/(n-1))*2-1;
      const noise=(Math.sin(i*37.3+pi*100)*0.5)*(1-Math.abs(panel.r));
      const y=panel.r*x+noise;
      pts.push([x,y]);
    }
    function toX(v){return ox+4+(v+1.5)/(3)*(pw-8);}
    function toY(v){return oy+4+(1-(v+1.5)/(3))*(ph-8);}
    const mx=pts.reduce((a,p)=>a+p[0],0)/n;
    const my=pts.reduce((a,p)=>a+p[1],0)/n;
    const denom=pts.reduce((a,p)=>a+(p[0]-mx)**2,1e-9);
    const slope=pts.reduce((a,p)=>a+(p[0]-mx)*(p[1]-my),0)/denom;
    const intercept=my-slope*mx;
    ctx.strokeStyle=panel.col; ctx.lineWidth=1; ctx.setLineDash([3,2]);
    ctx.beginPath(); ctx.moveTo(toX(-1),toY(-1*slope+intercept)); ctx.lineTo(toX(1),toY(1*slope+intercept)); ctx.stroke();
    ctx.setLineDash([]);
    pts.forEach(p=>{ ctx.fillStyle=panel.col+'bb'; ctx.beginPath(); ctx.arc(toX(p[0]),toY(p[1]),3,0,Math.PI*2); ctx.fill(); });
  });
  ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
  ctx.fillText('Pearson Correlation Coefficient',W/2,H-4);
""",

'shannon-entropy': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',BLU='#60a5fa';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:44,r:20,t:24,b:40};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  function Hfunc(p){if(p<=0||p>=1)return 0;return -p*Math.log2(p)-(1-p)*Math.log2(1-p);}
  function toX(p){return pad.l+p*cw;}
  function toY(h){return pad.t+ch-(h/1)*ch*0.85;}
  ctx.fillStyle='rgba(96,165,250,0.08)';
  ctx.beginPath(); ctx.moveTo(toX(0.01),pad.t+ch);
  for(let p=0.01;p<=0.99;p+=0.01){ctx.lineTo(toX(p),toY(Hfunc(p)));}
  ctx.lineTo(toX(0.99),pad.t+ch); ctx.closePath(); ctx.fill();
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  ctx.strokeStyle=BLU; ctx.lineWidth=2; ctx.beginPath();
  let first=true;
  for(let p=0.01;p<=0.99;p+=0.005){
    const x=toX(p),y=toY(Hfunc(p));
    first?(ctx.moveTo(x,y),first=false):ctx.lineTo(x,y);
  }
  ctx.stroke();
  ctx.strokeStyle='#fbbf2488'; ctx.lineWidth=1; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(toX(0.5),pad.t); ctx.lineTo(toX(0.5),toY(1)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l,toY(1)); ctx.lineTo(toX(0.5),toY(1)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#fbbf24'; ctx.font='bold 10px monospace'; ctx.textAlign='right';
  ctx.fillText('H=1 bit',pad.l-2,toY(1)+4);
  ctx.textAlign='center'; ctx.fillText('p=0.5',toX(0.5),pad.t+ch+13);
  ctx.fillStyle=TXT; ctx.font='9px monospace';
  [0,0.25,0.5,0.75,1].forEach(v=>ctx.fillText(v.toFixed(2),toX(v),pad.t+ch+27));
  ctx.textAlign='right';
  [0,0.5,1].forEach(v=>ctx.fillText(v.toFixed(1),pad.l-4,toY(v)+3));
  ctx.textAlign='center'; ctx.fillText('p (probability)',pad.l+cw/2,H-2);
""",

'kl-divergence': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',BLU='#60a5fa',ORG='#fb923c',PNK='#f472b6';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:40,r:20,t:28,b:36};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const xMin=-2,xMax=7;
  function gauss(x,mu,sig){return Math.exp(-0.5*((x-mu)/sig)**2)/(sig*Math.sqrt(2*Math.PI));}
  function toX(x){return pad.l+(x-xMin)/(xMax-xMin)*cw;}
  function toY(y,yMax){return pad.t+ch-(y/yMax)*ch*0.9;}
  const yMax=0.55;
  ctx.fillStyle='rgba(244,114,182,0.15)';
  ctx.beginPath();
  for(let x=xMin;x<=xMax;x+=0.05){
    const p=gauss(x,2,0.8),q=gauss(x,3,1.2);
    x===xMin?ctx.moveTo(toX(x),toY(Math.max(p,q),yMax)):ctx.lineTo(toX(x),toY(Math.max(p,q),yMax));
  }
  for(let x=xMax;x>=xMin;x-=0.05){
    ctx.lineTo(toX(x),toY(Math.min(gauss(x,2,0.8),gauss(x,3,1.2)),yMax));
  }
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  ctx.strokeStyle=BLU; ctx.lineWidth=2; ctx.beginPath();
  for(let x=xMin;x<=xMax;x+=0.05){const cx=toX(x),cy=toY(gauss(x,2,0.8),yMax);x===xMin?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);}
  ctx.stroke();
  ctx.strokeStyle=ORG; ctx.lineWidth=2; ctx.beginPath();
  for(let x=xMin;x<=xMax;x+=0.05){const cx=toX(x),cy=toY(gauss(x,3,1.2),yMax);x===xMin?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);}
  ctx.stroke();
  ctx.font='10px monospace'; ctx.textAlign='left';
  ctx.fillStyle=BLU; ctx.fillText('P (true, mu=2, s=0.8)',pad.l+4,pad.t+12);
  ctx.fillStyle=ORG; ctx.fillText('Q (approx, mu=3, s=1.2)',pad.l+4,pad.t+26);
  ctx.fillStyle=PNK; ctx.fillText('KL divergence area',pad.l+4,pad.t+40);
""",

'matrix-multiplication': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',BLU='#60a5fa',GRN='#4ade80',YEL='#fbbf24';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const n=3; const cellSz=28; const gap=18;
  const mw=n*cellSz;
  const ox=Math.floor((W-(mw*3+gap*2+32))/2); const oy=Math.floor((H-mw)/2)-10;
  function drawMatrix(ox2,oy2,hlRow,hlCol,label,borderCol){
    for(let r=0;r<n;r++) for(let c=0;c<n;c++){
      const x=ox2+c*cellSz, y=oy2+r*cellSz;
      let bg='#0f172a';
      if(hlRow!==null&&r===hlRow) bg='rgba(96,165,250,0.2)';
      if(hlCol!==null&&c===hlCol) bg='rgba(74,222,128,0.2)';
      if(hlRow!==null&&hlCol!==null&&r===hlRow&&c===hlCol) bg='rgba(251,191,36,0.35)';
      ctx.fillStyle=bg; ctx.fillRect(x+1,y+1,cellSz-2,cellSz-2);
      ctx.strokeStyle=AXES; ctx.lineWidth=1; ctx.strokeRect(x,y,cellSz,cellSz);
      ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
      ctx.fillText(r*n+c+1,x+cellSz/2,y+cellSz/2+4);
    }
    if(borderCol){ctx.strokeStyle=borderCol;ctx.lineWidth=2;ctx.strokeRect(ox2,oy2,mw,mw);}
    ctx.fillStyle=TXT; ctx.font='bold 11px monospace'; ctx.textAlign='center';
    ctx.fillText(label,ox2+mw/2,oy2-6);
  }
  const ax=ox, bx=ox+mw+gap+16, cx2=bx+mw+gap+16;
  drawMatrix(ax,oy,0,null,'A',BLU);
  drawMatrix(bx,oy,null,0,'B',GRN);
  drawMatrix(cx2,oy,0,0,'C',YEL);
  ctx.fillStyle='#94a3b8'; ctx.font='bold 16px sans-serif'; ctx.textAlign='center';
  ctx.fillText('x',ox+mw+gap/2+8,oy+mw/2+6);
  ctx.fillText('=',bx+mw+gap/2+8,oy+mw/2+6);
  ctx.fillStyle=YEL; ctx.font='10px monospace'; ctx.textAlign='center';
  ctx.fillText('C[i][j] = sum_k A[i][k]*B[k][j]',W/2,oy+mw+22);
""",

'adam-implementation': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',RED='#f87171',ORG='#fb923c',GRN='#4ade80';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:44,r:20,t:28,b:36};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const N=100;
  function sgdLoss(i){return 0.9*Math.exp(-i/120)+0.1+Math.sin(i*0.8)*0.06*(1-i/150);}
  function momLoss(i){return 0.85*Math.exp(-i/80)+0.08+Math.sin(i*0.4)*0.025*(1-i/120);}
  function adamLoss(i){return 0.8*Math.exp(-i/35)+0.05+Math.sin(i*0.2)*0.01;}
  function toX(i){return pad.l+(i/N)*cw;}
  function toY(v){return pad.t+ch-Math.min(1,Math.max(0,(v/1.05)*ch*0.88));}
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  [[sgdLoss,RED,'SGD'],[momLoss,ORG,'Momentum'],[adamLoss,GRN,'Adam']].forEach(([fn,col,lbl],li)=>{
    ctx.strokeStyle=col; ctx.lineWidth=2; ctx.beginPath();
    for(let i=0;i<=N;i++){const x=toX(i),y=toY(fn(i));i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
    ctx.stroke();
    ctx.fillStyle=col; ctx.font='10px monospace'; ctx.textAlign='right';
    ctx.fillText(lbl,pad.l+cw-4,pad.t+16+li*14);
  });
  ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
  ctx.fillText('Iterations (0-100)',pad.l+cw/2,H-4);
  [0,0.5,1].forEach(v=>{ctx.textAlign='right';ctx.fillText(v.toFixed(1),pad.l-4,toY(v)+3);});
  ctx.save(); ctx.translate(12,pad.t+ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign='center'; ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.fillText('Loss',0,0); ctx.restore();
""",

'taylor-approximation': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',WHT='#e2e8f0',RED='#f87171',ORG='#fb923c',GRN='#4ade80';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:40,r:20,t:24,b:36};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const xMin=-Math.PI,xMax=Math.PI,yMin=-1.5,yMax=1.5;
  function toX(x){return pad.l+(x-xMin)/(xMax-xMin)*cw;}
  function toY(y){return pad.t+ch-(y-yMin)/(yMax-yMin)*ch;}
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,toY(0)); ctx.lineTo(pad.l+cw,toY(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(toX(0),pad.t); ctx.lineTo(toX(0),pad.t+ch); ctx.stroke();
  function drawFn(fn,col,dash){
    ctx.strokeStyle=col; ctx.lineWidth=1.5;
    if(dash) ctx.setLineDash(dash); else ctx.setLineDash([]);
    ctx.beginPath(); let first=true;
    for(let x=xMin;x<=xMax;x+=0.03){
      const y=fn(x); if(isNaN(y)||!isFinite(y)) continue;
      const cy=toY(Math.max(yMin,Math.min(yMax,y)));
      first?(ctx.moveTo(toX(x),cy),first=false):ctx.lineTo(toX(x),cy);
    }
    ctx.stroke(); ctx.setLineDash([]);
  }
  drawFn(x=>Math.sin(x),WHT);
  drawFn(x=>x,RED,[4,3]);
  drawFn(x=>x-x**3/6,ORG,[4,3]);
  drawFn(x=>x-x**3/6+x**5/120,GRN,[4,3]);
  ctx.font='9px monospace'; ctx.textAlign='left';
  [[WHT,'sin(x)'],[RED,'deg 1: x'],[ORG,'deg 3'],[GRN,'deg 5']].forEach(([c,l],i)=>{
    ctx.fillStyle=c; ctx.fillText(l,pad.l+4,pad.t+12+i*12);
  });
  ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
  ctx.fillText('Taylor Approximations of sin(x)',W/2,H-4);
""",

'convexity-check': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',GRN='#4ade80',RED='#f87171';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const half=Math.floor(W/2)-4;
  const pad={l:20,r:10,t:28,b:36};
  function drawPanel(ox,ow,fn,col,label){
    const ch=H-pad.t-pad.b;
    const xMin=-2,xMax=2;
    const vals=[]; for(let x=xMin;x<=xMax;x+=0.1) vals.push(fn(x));
    const yMin2=Math.min(...vals)-0.2, yMax2=Math.max(...vals)+0.2;
    function toX(x){return ox+pad.l+(x-xMin)/(xMax-xMin)*(ow-pad.l-pad.r);}
    function toY(y){return pad.t+ch-(y-yMin2)/(yMax2-yMin2)*ch;}
    ctx.strokeStyle=AXES; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(ox+pad.l,pad.t); ctx.lineTo(ox+pad.l,pad.t+ch); ctx.lineTo(ox+ow-pad.r,pad.t+ch); ctx.stroke();
    ctx.strokeStyle=col; ctx.lineWidth=2; ctx.beginPath(); let first=true;
    for(let x=xMin;x<=xMax;x+=0.04){
      const y=fn(x); const cx=toX(x),cy=toY(y);
      first?(ctx.moveTo(cx,cy),first=false):ctx.lineTo(cx,cy);
    }
    ctx.stroke();
    const x1=-1.5,x2=1.5;
    ctx.strokeStyle=col+'88'; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(toX(x1),toY(fn(x1))); ctx.lineTo(toX(x2),toY(fn(x2))); ctx.stroke();
    ctx.setLineDash([]);
    [x1,x2].forEach(x=>{ctx.fillStyle=col;ctx.beginPath();ctx.arc(toX(x),toY(fn(x)),4,0,Math.PI*2);ctx.fill();});
    ctx.fillStyle=col; ctx.font='bold 10px monospace'; ctx.textAlign='center';
    ctx.fillText(label,ox+ow/2,pad.t-8);
  }
  drawPanel(0,half,x=>x*x+0.1,GRN,'Convex (check)');
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(half+4,0); ctx.lineTo(half+4,H); ctx.stroke();
  drawPanel(half+8,half,x=>Math.sin(x*2)*0.4+x*x*0.1,RED,'Non-Convex (x)');
  ctx.fillStyle=TXT; ctx.font='9px monospace'; ctx.textAlign='center';
  ctx.fillText('chord above = convex',W/4,H-4);
  ctx.fillText('chord below = non-convex',W*3/4,H-4);
""",

'gradient-computation': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',RED='#f87171',GRN='#4ade80';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const cx=Math.floor(W/2), cy=Math.floor(H/2);
  const colors=['#1e3a5f','#1e4080','#1e4d99','#1a5db3','#1a6bcc'];
  for(let i=0;i<5;i++){
    const rx=(i+1)*28, ry=(i+1)*18;
    ctx.strokeStyle=colors[i]; ctx.lineWidth=1;
    ctx.beginPath(); ctx.ellipse(cx,cy,rx,ry,0,0,Math.PI*2); ctx.stroke();
  }
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(30,cy); ctx.lineTo(W-20,cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx,20); ctx.lineTo(cx,H-20); ctx.stroke();
  const scaleX=36, scaleY=24;
  const px=cx+1.5*scaleX, py=cy-1.0*scaleY;
  ctx.fillStyle=RED; ctx.beginPath(); ctx.arc(px,py,5,0,Math.PI*2); ctx.fill();
  const gx=3,gy=2; const glen=Math.sqrt(gx*gx+gy*gy);
  const arrowLen=45;
  const dx=(gx/glen)*arrowLen, dy=-(gy/glen)*arrowLen;
  ctx.strokeStyle=GRN; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px+dx,py+dy); ctx.stroke();
  const angle=Math.atan2(dy,dx);
  ctx.fillStyle=GRN; ctx.beginPath();
  ctx.moveTo(px+dx,py+dy);
  ctx.lineTo(px+dx-10*Math.cos(angle-0.4),py+dy-10*Math.sin(angle-0.4));
  ctx.lineTo(px+dx-10*Math.cos(angle+0.4),py+dy-10*Math.sin(angle+0.4));
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle='#f472b6'; ctx.lineWidth=1.5; ctx.setLineDash([3,2]);
  ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(px-dx*0.7,py-dy*0.7); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle=GRN; ctx.font='9px monospace'; ctx.textAlign='left';
  ctx.fillText('grad-f (uphill)',px+dx+4,py+dy);
  ctx.fillStyle='#f472b6'; ctx.textAlign='right';
  ctx.fillText('-grad-f (descent)',px-dx*0.7-4,py-dy*0.7);
  ctx.fillStyle=TXT; ctx.textAlign='center';
  ctx.fillText('f(x,y) = x^2 + y^2  contours',W/2,H-8);
""",

'sgd-minibatch': """\
  const BG='#0a0a0f',AXES='#334155',TXT='#64748b',ORG='#fb923c',GRN='#4ade80';
  ctx.fillStyle=BG; ctx.fillRect(0,0,W,H);
  const pad={l:40,r:24,t:28,b:48};
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  function lossFn(x){return (x-0.5)**2*0.8+0.05+Math.sin(x*18)*0.015+Math.cos(x*11)*0.012;}
  function toX(x){return pad.l+x*cw;}
  function toY(v,yMin,yMax){return pad.t+ch-(v-yMin)/(yMax-yMin)*ch;}
  const steps=15; let wx=0.1;
  const lr=0.12; const pts=[{x:wx,y:lossFn(wx)}];
  for(let i=0;i<steps-1;i++){
    const grad=(lossFn(wx+0.001)-lossFn(wx-0.001))/0.002+Math.sin(i*3.7)*0.04;
    wx=Math.max(0.02,Math.min(0.98,wx-lr*grad));
    pts.push({x:wx,y:lossFn(wx)});
  }
  const yMin=0, yMax=Math.max(...pts.map(p=>p.y))*1.3;
  ctx.strokeStyle=AXES; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch); ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();
  ctx.strokeStyle='#334155'; ctx.lineWidth=2; ctx.beginPath();
  for(let x=0;x<=1;x+=0.005){
    const cx=toX(x),cy=toY(lossFn(x),yMin,yMax);
    x===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
  }
  ctx.stroke();
  for(let i=0;i<pts.length-1;i++){
    const p1=pts[i],p2=pts[i+1];
    ctx.strokeStyle=ORG+'88'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(toX(p1.x),toY(p1.y,yMin,yMax)); ctx.lineTo(toX(p2.x),toY(p2.y,yMin,yMax)); ctx.stroke();
    ctx.fillStyle=ORG; ctx.beginPath(); ctx.arc(toX(p1.x),toY(p1.y,yMin,yMax),3,0,Math.PI*2); ctx.fill();
  }
  const last=pts[pts.length-1];
  ctx.fillStyle=GRN; ctx.beginPath(); ctx.arc(toX(last.x),toY(last.y,yMin,yMax),5,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='#fbbf24'; ctx.font='10px monospace'; ctx.textAlign='center';
  ctx.fillText('w = w - lr * grad_L',W/2,H-6);
  ctx.fillStyle=TXT; ctx.font='9px monospace';
  ctx.fillText('SGD steps: orange=path, green=minimum',W/2,H-18);
""",

}  # end VISUALIZATIONS

# -- Build VISUALIZATIONS JS object ------------------------------------------
def build_viz_js():
    lines = ['const VISUALIZATIONS = {']
    for slug, body in VISUALIZATIONS.items():
        lines.append(f"  '{slug}': (canvas,W,H,ctx)=>{{")  
        lines.append(body.rstrip('\n'))
        lines.append('  },')
    lines.append('};')
    return '\n'.join(lines)



# ── Build JS data objects ────────────────────────────────────────────────────
lessons_js_entries = []
for slug, data in lessons.items():
    entry = {
        'slug':         slug,
        'title':        data['title'],
        'section':      data['section'],
        'topic':        data['topic'],
        'description':  data['description'],
        'url':          data['url'],
        'html_content': data.get('html_content', ''),
    }
    lessons_js_entries.append(entry)

lessons_js = 'const LESSONS = ' + json.dumps(
    {e['slug']: e for e in lessons_js_entries},
    ensure_ascii=False, indent=None
) + ';'

# ── HTML ─────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CUDA × ML Math Roadmap</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>">
  <style>
    :root {
      --bg: #0f172a; --surface: #1e293b; --surf2: #263347;
      --text: #e2e8f0; --muted: #64748b; --border: #334155; --done: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

    header { position: sticky; top: 0; z-index: 50; background: rgba(15,23,42,.96); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border); padding: 14px 28px; flex-shrink: 0; }
    .hrow { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .htitle { font-size: 1.3rem; font-weight: 700; display: flex; align-items: center; gap: 10px; }
    .hsub { font-size: .7rem; color: var(--muted); border: 1px solid var(--border); border-radius: 999px; padding: 2px 8px; font-weight: 400; }
    .hprog { font-size: .82rem; color: var(--muted); font-variant-numeric: tabular-nums; }
    .ptrack { height: 4px; background: var(--border); border-radius: 999px; overflow: hidden; }
    .pfill { height: 100%; width: 0%; background: linear-gradient(90deg,#10b981,#06b6d4,#8b5cf6); border-radius: 999px; transition: width .5s; }

    .app-body { display: flex; flex: 1; overflow: hidden; }
    #roadmap-scroll { flex: 1; overflow-y: auto; transition: flex 0.3s ease; }
    .app-body.panel-open #roadmap-scroll { flex: 0 0 54%; }

    #wrapper { max-width: 960px; margin: 0 auto; padding: 48px 28px 100px; position: relative; }
    #lines { position: absolute; top: 0; left: 0; width: 100%; pointer-events: none; overflow: visible; }

    .section { margin-bottom: 72px; }
    .phase-node { display: flex; flex-direction: column; align-items: center; text-align: center; padding: 20px 36px; border-radius: 16px; border: 2px solid; margin: 0 auto; max-width: 440px; position: relative; z-index: 2; }
    .pnum { font-size: .65rem; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 4px; opacity: .7; }
    .pname { font-size: 1.15rem; font-weight: 700; margin-bottom: 4px; }
    .psub { font-size: .72rem; opacity: .65; }
    .pcount { font-size: .72rem; margin-top: 6px; opacity: .5; }

    .topic-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 40px; padding: 0 12px; }

    .topic { display: flex; align-items: center; gap: 8px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 9px 14px; font-size: .81rem; font-weight: 500; user-select: none; position: relative; z-index: 2; transition: background .15s, transform .15s, box-shadow .15s, border-color .2s; }
    .topic.has-lesson { cursor: pointer; }
    .topic.has-lesson:hover { background: var(--surf2); transform: translateY(-2px); box-shadow: 0 5px 18px rgba(0,0,0,.4); }
    .topic.done { border-color: rgba(16,185,129,.45); background: rgba(16,185,129,.07); color: var(--muted); text-decoration: line-through; text-decoration-color: rgba(16,185,129,.5); }
    .topic.selected { outline: 2px solid #60a5fa; outline-offset: 2px; }
    .topic.pop { animation: pop .25s ease; }
    @keyframes pop { 0%{transform:scale(1)} 45%{transform:scale(1.08)} 100%{transform:scale(1)} }
    .dot { width: 16px; height: 16px; border-radius: 50%; border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; font-size: 9px; color: transparent; flex-shrink: 0; transition: all .2s; cursor: pointer; }
    .dot:hover { border-color: var(--done); }
    .topic.done .dot { background: var(--done); border-color: var(--done); color: #fff; }

    .reset-btn { display: block; margin: 24px auto 0; padding: 8px 22px; background: transparent; border: 1px solid var(--border); border-radius: 6px; color: var(--muted); font-size: .78rem; cursor: pointer; transition: all .2s; }
    .reset-btn:hover { border-color: #ef4444; color: #ef4444; }

    #lesson-pane { width: 0; overflow: hidden; border-left: 1px solid var(--border); background: #0d1117; display: flex; flex-direction: column; transition: width 0.3s ease; flex-shrink: 0; }
    .app-body.panel-open #lesson-pane { width: 46%; }
    .lesson-header { display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
    .lesson-header-info { flex: 1; min-width: 0; }
    .lesson-header h2 { font-size: .9rem; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .lesson-section-badge { font-size: .65rem; padding: 2px 8px; border-radius: 4px; background: var(--surface); color: var(--muted); margin-top: 3px; display: inline-block; }
    .close-btn { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 1.1rem; padding: 4px 8px; border-radius: 4px; flex-shrink: 0; }
    .close-btn:hover { background: var(--surface); color: var(--text); }

    .tabs { display: flex; border-bottom: 1px solid var(--border); flex-shrink: 0; }
    .tab { padding: 10px 18px; font-size: .8rem; font-weight: 600; cursor: pointer; color: var(--muted); border-bottom: 2px solid transparent; transition: color .15s; }
    .tab:hover { color: #94a3b8; }
    .tab.active { color: #60a5fa; border-bottom-color: #60a5fa; }
    .tab-link { margin-left: auto; padding: 10px 14px; font-size: .75rem; color: var(--border); text-decoration: none; }
    .tab-link:hover { color: #60a5fa; }
    .tab-content { flex: 1; overflow-y: auto; display: none; }
    .tab-content.active { display: block; }

    .theory-body { padding: 18px; font-size: .82rem; line-height: 1.75; color: #94a3b8; }
    .theory-body p { margin-bottom: 10px; }
    .def-box { background: var(--bg); border-left: 3px solid #60a5fa; padding: 10px 14px; border-radius: 4px; margin: 10px 0; font-size: .8rem; }
    .ex-box  { background: var(--bg); border-left: 3px solid #4ade80; padding: 10px 14px; border-radius: 4px; margin: 10px 0; font-size: .8rem; }
    .section-hdr { color: var(--text); font-weight: 700; font-size: .85rem; margin: 14px 0 6px; }
    .theory-body strong { color: #cbd5e1; }
    .theory-title { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
    .theory-topic { font-size: .68rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: var(--border); margin-bottom: 14px; }

    .code-pane { padding: 14px; display: flex; flex-direction: column; gap: 10px; }
    .code-toolbar { display: flex; align-items: center; gap: 8px; }
    .code-filename { font-size: .72rem; color: var(--muted); font-family: monospace; background: var(--bg); padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border); }
    .copy-btn { margin-left: auto; background: var(--surface); border: 1px solid var(--border); color: #94a3b8; font-size: .72rem; padding: 4px 12px; border-radius: 4px; cursor: pointer; }
    .copy-btn:hover { background: var(--border); color: var(--text); }
    .copy-btn.copied { background: #14532d33; color: #4ade80; border-color: #166534; }
    pre { background: #060609; border: 1px solid var(--border); border-radius: 8px; padding: 16px; overflow-x: auto; font-size: .75rem; line-height: 1.6; font-family: 'Cascadia Code','Fira Code',monospace; color: #94a3b8; white-space: pre; }
    .kw{color:#c792ea;} .ty{color:#82aaff;} .fn{color:#82aaff;} .cm{color:#546e7a;font-style:italic;} .nu{color:#f78c6c;} .pp{color:#c792ea;} .deco{color:#ffcb6b;}
    .lesson-footer { padding: 12px 18px; border-top: 1px solid var(--border); display: flex; gap: 10px; align-items: center; flex-shrink: 0; }
    .done-btn { flex: 1; padding: 8px; border-radius: 8px; font-size: .8rem; font-weight: 600; cursor: pointer; border: 1px solid #166534; background: #14532d33; color: #4ade80; transition: background .15s; }
    .done-btn:hover { background: #14532d66; }
    .done-btn.already-done { background: #14532d; color: #4ade80; cursor: default; }
    #viz-canvas { width: 100%; max-height: 220px; display: none; border-bottom: 1px solid var(--border); background: #0a0a0f; }
  </style>
</head>
<body>

<header>
  <div class="hrow">
    <div class="htitle">⚡ CUDA × ML Math <span class="hsub">RTX 3060 · CUDA 12.8</span></div>
    <span class="hprog" id="prog-text">0 / 0 completed</span>
  </div>
  <div class="ptrack"><div class="pfill" id="prog-fill"></div></div>
</header>

<div class="app-body" id="app-body">
  <div id="roadmap-scroll">
    <div id="wrapper">
      <svg id="lines"></svg>
      <div id="graph"></div>
    </div>
  </div>

  <div id="lesson-pane">
    <div class="lesson-header">
      <div class="lesson-header-info">
        <h2 id="lp-title">—</h2>
        <span class="lesson-section-badge" id="lp-section"></span>
      </div>
      <button class="close-btn" onclick="closePanel()">✕</button>
    </div>
    <div class="tabs">
      <a id="lp-link" href="#" target="_blank" class="tab-link">↗ tensortonic</a>
    </div>
    <div id="tab-theory" class="tab-content active">
      <div id="lp-viz-html" style="display:none;border-bottom:1px solid var(--border);"></div>
      <canvas id="viz-canvas" width="480" height="220"></canvas>
      <div class="theory-body">
        <div id="lp-description"></div>
      </div>
    </div>
    <div class="lesson-footer">
      <button class="done-btn" id="done-btn" onclick="markDone()">✓ Mark as Done</button>
    </div>
  </div>
</div>

<script>
LESSONS_PLACEHOLDER

const PHASES = [
  {n:1, title:'GPU Foundations', color:'#10b981', sub:'Thread model · first kernels · memory model', topics:[
    {name:'CUDA Kernel Foundations', problems:[
      {t:'Vector Addition',    s:'done', slug:'vector-addition'},
      {t:'Vector Subtraction', s:'done', slug:'vector-subtraction'},
      {t:'ReLU',               s:'done', slug:'relu'},
      {t:'Sigmoid',            s:'done', slug:'sigmoid'},
      {t:'Tanh',               s:'done', slug:'tanh'},
      {t:'Leaky ReLU',         s:'done', slug:'leaky-relu'},
      {t:'GELU',               s:'done', slug:'gelu'},
      {t:'Swish',              s:'done', slug:'swish'},
    ]},
    {name:'CUDA Concepts', problems:[
      {t:'Thread / Block / Grid', s:'done',   slug:'thread-block-grid'},
      {t:'2D / 3D Index Formula', s:'done',   slug:'index-2d-3d'},
      {t:'Memory Coalescing',     s:'locked', slug:'memory-coalescing'},
      {t:'Shared Memory',         s:'locked', slug:null},
    ]},
  ]},
  {n:2, title:'Statistics on GPU', color:'#3b82f6', sub:'Parallel reduction → statistical kernels', topics:[
    {name:'Descriptive Statistics', problems:[
      {t:'Calculate Mean',        s:'locked', slug:'calculate-mean'},
      {t:'Calculate Variance',    s:'locked', slug:'calculate-variance-std'},
      {t:'Population vs Sample',  s:'locked', slug:'population-sample-stats'},
    ]},
    {name:'Sampling & Inference', problems:[
      {t:'Standard Error',        s:'locked', slug:'standard-error-calculation'},
      {t:'Central Limit Theorem', s:'locked', slug:'clt-simulation'},
      {t:'Confidence Intervals',  s:'locked', slug:'ci-mean-known-sigma'},
    ]},
    {name:'Hypothesis Testing', problems:[
      {t:'Hypothesis Setup',      s:'locked', slug:'hypothesis-setup'},
      {t:'P-Value from Z',        s:'locked', slug:'p-value-from-z'},
      {t:'T-Test Statistic',      s:'locked', slug:'t-test-statistic'},
      {t:'A/B Test Setup',        s:'locked', slug:'ab-test-setup'},
    ]},
    {name:'Correlation & MLE', problems:[
      {t:'Pearson Correlation',   s:'locked', slug:'pearson-correlation'},
      {t:'MLE Bernoulli',         s:'locked', slug:'mle-bernoulli'},
    ]},
  ]},
  {n:3, title:'Linear Algebra on GPU', color:'#8b5cf6', sub:'Matrix kernels · decompositions · geometric ops', topics:[
    {name:'Matrix Operations', problems:[
      {t:'Matrix Multiplication', s:'locked', slug:'matrix-multiplication'},
      {t:'Vector Norms',          s:'locked', slug:'vector-norms'},
      {t:'Gram-Schmidt',          s:'locked', slug:'gram-schmidt'},
    ]},
    {name:'Decompositions', problems:[
      {t:'Eigenvalue Analysis',   s:'locked', slug:'eigenvalue-analysis'},
      {t:'SVD Decomposition',     s:'locked', slug:'svd-decomposition'},
      {t:'PCA from Scratch',      s:'locked', slug:'pca-from-scratch'},
    ]},
  ]},
  {n:4, title:'Probability on GPU', color:'#14b8a6', sub:'Sampling · Monte Carlo · Bayesian kernels', topics:[
    {name:'Probability Kernels', problems:[
      {t:'Conditional Probability',   s:'locked', slug:'conditional-probability'},
      {t:'PMF / PDF / CDF',           s:'locked', slug:'pmf-pdf-cdf'},
      {t:'Expected Value & Variance', s:'locked', slug:'expected-value-variance'},
      {t:'Bayes Theorem',             s:'locked', slug:'bayes-theorem'},
      {t:'Monte Carlo Pi',            s:'locked', slug:'monte-carlo-pi'},
    ]},
  ]},
  {n:5, title:'Calculus & Autograd', color:'#f97316', sub:'Gradients · backprop · numerical differentiation', topics:[
    {name:'Calculus Kernels', problems:[
      {t:'Numerical Limits',     s:'locked', slug:'numerical-limits'},
      {t:'Gradient Computation', s:'locked', slug:'gradient-computation'},
      {t:'Chain Rule Backprop',  s:'locked', slug:'chain-rule-backprop'},
      {t:'Hessian Computation',  s:'locked', slug:'hessian-computation'},
      {t:'Taylor Approximation', s:'locked', slug:'taylor-approximation'},
      {t:'Manual Backprop',      s:'locked', slug:'manual-backprop'},
    ]},
  ]},
  {n:6, title:'Optimization on GPU', color:'#ec4899', sub:'SGD · Adam · regularization kernels', topics:[
    {name:'Optimizer Kernels', problems:[
      {t:'Convexity Check',      s:'locked', slug:'convexity-check'},
      {t:'SGD Mini-Batch',       s:'locked', slug:'sgd-minibatch'},
      {t:'Momentum Optimizer',   s:'locked', slug:'momentum-optimizer'},
      {t:'Adam Optimizer',       s:'locked', slug:'adam-implementation'},
      {t:'L1/L2 Regularization', s:'locked', slug:'l1-l2-regularization'},
    ]},
  ]},
  {n:7, title:'Information Theory on GPU', color:'#eab308', sub:'Entropy · KL divergence · mutual information', topics:[
    {name:'Information Theory Kernels', problems:[
      {t:'Shannon Entropy',   s:'locked', slug:'shannon-entropy'},
      {t:'Cross-Entropy',     s:'locked', slug:'cross-entropy-implementation'},
      {t:'KL Divergence',     s:'locked', slug:'kl-divergence'},
      {t:'Mutual Information',s:'locked', slug:'mutual-information'},
      {t:'Information Gain',  s:'locked', slug:'information-gain'},
    ]},
  ]},
  {n:8, title:'Performance & Profiling', color:'#06b6d4', sub:'Tiling · shared memory · ncu · occupancy', topics:[
    {name:'Performance', problems:[
      {t:'Tiled Matrix Multiplication', s:'locked', slug:null},
      {t:'Warp Divergence Analysis',    s:'locked', slug:null},
      {t:'Memory Access Patterns',      s:'locked', slug:null},
      {t:'ncu Profiling',               s:'locked', slug:null},
    ]},
  ]},
];

const KEY = 'cuda-roadmap-v2';
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch {}
if (!localStorage.getItem(KEY)) {
  PHASES.forEach(ph => ph.topics.forEach(tg => tg.problems.forEach(p => {
    if (p.s === 'done') saved[p.slug || p.t] = true;
  })));
}
const persist = () => localStorage.setItem(KEY, JSON.stringify(saved));

let currentSlug = null;
let currentChipEl = null;

function updateProgress() {
  const allP = PHASES.flatMap(p => p.topics.flatMap(t => t.problems));
  const total = allP.length;
  const doneCount = allP.filter(p => saved[p.slug || p.t]).length;
  document.getElementById('prog-text').textContent = `${doneCount} / ${total} completed`;
  document.getElementById('prog-fill').style.width = (total ? doneCount/total*100 : 0) + '%';
  PHASES.forEach(p => {
    const problems = p.topics.flatMap(t => t.problems);
    const n = problems.filter(prob => saved[prob.slug || prob.t]).length;
    const el = document.getElementById(`cnt-${p.n}`);
    if (el) el.textContent = `${n} / ${problems.length}`;
  });
}

function toggleDone(key, chipEl) {
  saved[key] = !saved[key];
  chipEl.classList.toggle('done', !!saved[key]);
  chipEl.classList.remove('pop');
  void chipEl.offsetWidth;
  chipEl.classList.add('pop');
  persist();
  updateProgress();
}

function render() {
  const graph = document.getElementById('graph');
  graph.innerHTML = '';

  PHASES.forEach(p => {
    const allProblems = p.topics.flatMap(t => t.problems);
    const sec = document.createElement('div');
    sec.className = 'section';

    const pNode = document.createElement('div');
    pNode.className = 'phase-node';
    pNode.id = `pn-${p.n}`;
    pNode.style.borderColor = p.color + '55';
    pNode.style.background  = p.color + '0e';
    pNode.innerHTML = `
      <div class="pnum" style="color:${p.color}">Phase ${p.n}</div>
      <div class="pname">${p.title}</div>
      <div class="psub" style="color:${p.color}99">${p.sub}</div>
      <div class="pcount" id="cnt-${p.n}">0 / ${allProblems.length}</div>`;
    sec.appendChild(pNode);

    const row = document.createElement('div');
    row.className = 'topic-row';
    row.id = `tr-${p.n}`;

    allProblems.forEach((prob, i) => {
      const key = prob.slug || prob.t;
      const isDone = !!saved[key];
      const hasLesson = !!(prob.slug && LESSONS[prob.slug]);

      const el = document.createElement('div');
      el.className = 'topic' + (isDone ? ' done' : '') + (hasLesson ? ' has-lesson' : '');
      el.id = `tn-${p.n}-${i}`;
      el.dataset.key = key;
      el.dataset.slug = prob.slug || '';
      el.innerHTML = `<div class="dot">✓</div><span>${prob.t}</span>`;

      el.querySelector('.dot').addEventListener('click', e => {
        e.stopPropagation();
        toggleDone(key, el);
      });
      if (hasLesson) {
        el.addEventListener('click', () => openLesson(prob.slug, el));
      }

      row.appendChild(el);
    });

    sec.appendChild(row);
    graph.appendChild(sec);
  });

  const btn = document.createElement('button');
  btn.className = 'reset-btn';
  btn.textContent = '↺ Reset Progress';
  btn.addEventListener('click', () => {
    if (!confirm('Reset all progress?')) return;
    saved = {};
    PHASES.forEach(ph => ph.topics.forEach(tg => tg.problems.forEach(p => {
      if (p.s === 'done') saved[p.slug || p.t] = true;
    })));
    persist();
    document.querySelectorAll('.topic').forEach(el =>
      el.classList.toggle('done', !!saved[el.dataset.key]));
    updateProgress();
    drawLines();
  });
  graph.appendChild(btn);
}

function drawLines() {
  const svg     = document.getElementById('lines');
  const wrapper = document.getElementById('wrapper');
  svg.innerHTML = '';

  const wRect = wrapper.getBoundingClientRect();

  function rr(el) {
    const r = el.getBoundingClientRect();
    return {
      top:    r.top    - wRect.top,
      bottom: r.bottom - wRect.top,
      cx:    (r.left + r.right) / 2 - wRect.left,
    };
  }

  function line(x1,y1,x2,y2,stroke,sw,dash) {
    const el = document.createElementNS('http://www.w3.org/2000/svg','line');
    el.setAttribute('x1',x1); el.setAttribute('y1',y1);
    el.setAttribute('x2',x2); el.setAttribute('y2',y2);
    el.setAttribute('stroke',stroke); el.setAttribute('stroke-width',sw||2);
    if (dash) el.setAttribute('stroke-dasharray',dash);
    svg.appendChild(el);
  }

  function bezier(x1,y1,x2,y2,stroke,sw) {
    const path = document.createElementNS('http://www.w3.org/2000/svg','path');
    const my = (y1+y2)/2;
    path.setAttribute('d',`M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`);
    path.setAttribute('fill','none');
    path.setAttribute('stroke',stroke);
    path.setAttribute('stroke-width',sw||1.5);
    svg.appendChild(path);
  }

  PHASES.forEach((p, pi) => {
    const phaseEl = document.getElementById(`pn-${p.n}`);
    const rowEl   = document.getElementById(`tr-${p.n}`);
    if (!phaseEl || !rowEl) return;

    const pR = rr(phaseEl);
    const junctionY = pR.bottom + 18;
    line(pR.cx, pR.bottom, pR.cx, junctionY, p.color + '55', 2);

    rowEl.querySelectorAll('.topic').forEach(topic => {
      const tR = rr(topic);
      bezier(pR.cx, junctionY, tR.cx, tR.top, p.color + '45', 1.5);
    });

    if (pi < PHASES.length - 1) {
      const nextEl = document.getElementById(`pn-${PHASES[pi+1].n}`);
      if (nextEl) {
        const nR   = rr(nextEl);
        const rowR = rr(rowEl);
        line(pR.cx, rowR.bottom + 8, nR.cx, nR.top, '#334155', 2, '6,5');
      }
    }
  });

  svg.setAttribute('height', wrapper.scrollHeight);
}


function openLesson(slug, chipEl) {
  const lesson = LESSONS[slug];
  if (!lesson) return;

  if (currentChipEl) currentChipEl.classList.remove('selected');
  currentSlug   = slug;
  currentChipEl = chipEl;
  if (chipEl) chipEl.classList.add('selected');

  document.getElementById('lp-title').textContent        = lesson.title;
  document.getElementById('lp-section').textContent      = lesson.section;
  document.getElementById('lp-link').href                = lesson.url;

  function formatDescription(text) {
    return text.split(/\n{2,}|(?<=[.!?])\s{2,}(?=[A-Z])/)
      .map(p => p.trim()).filter(p => p.length > 8).slice(0, 18)
      .map(chunk => {
        if (/^Definition[:\s]/i.test(chunk)) return `<div class="def-box">${chunk}</div>`;
        if (/^Example[:\s]/i.test(chunk))    return `<div class="ex-box">${chunk}</div>`;
        return `<p>${chunk}</p>`;
      }).join('');
  }
  document.getElementById('lp-description').innerHTML = formatDescription(lesson.description);

  const isDone = !!saved[slug];
  const btn = document.getElementById('done-btn');
  btn.textContent = isDone ? '✓ Already Done' : '✓ Mark as Done';
  btn.className   = 'done-btn' + (isDone ? ' already-done' : '');


  document.getElementById('app-body').classList.add('panel-open');

  const vizHtml   = document.getElementById('lp-viz-html');
  const vizCanvas = document.getElementById('viz-canvas');
  if (lesson.html_content) {
    vizHtml.innerHTML = lesson.html_content;
    vizHtml.style.display = 'block';
    vizCanvas.style.display = 'none';
  } else if (VISUALIZATIONS && VISUALIZATIONS[slug]) {
    vizHtml.style.display = 'none';
    vizCanvas.style.display = 'block';
    const ctx = vizCanvas.getContext('2d');
    ctx.clearRect(0, 0, vizCanvas.width, vizCanvas.height);
    try { VISUALIZATIONS[slug](vizCanvas, vizCanvas.width, vizCanvas.height, ctx); }
    catch(e) { console.warn('viz error', slug, e); }
  } else {
    vizHtml.style.display = 'none';
    vizCanvas.style.display = 'none';
  }
}

function closePanel() {
  document.getElementById('app-body').classList.remove('panel-open');
  if (currentChipEl) currentChipEl.classList.remove('selected');
  currentSlug = null; currentChipEl = null;
}



function markDone() {
  if (!currentSlug || saved[currentSlug]) return;
  saved[currentSlug] = true;
  persist();
  const btn = document.getElementById('done-btn');
  btn.textContent = '✓ Already Done'; btn.className = 'done-btn already-done';
  const chip = document.querySelector(`[data-slug="${currentSlug}"]`);
  if (chip) { chip.classList.add('done'); currentChipEl = chip; }
  updateProgress();
}

VISUALIZATIONS_PLACEHOLDER
render();
updateProgress();
requestAnimationFrame(() => requestAnimationFrame(drawLines));
window.addEventListener('resize', () => requestAnimationFrame(drawLines));
document.getElementById('lesson-pane').addEventListener('transitionend', drawLines);
</script>
</body>
</html>
"""

# Inject lessons data and visualizations
viz_js = build_viz_js()
final_html = HTML.replace('LESSONS_PLACEHOLDER', lessons_js)
final_html = final_html.replace('VISUALIZATIONS_PLACEHOLDER', viz_js)

OUT_FILE.write_text(final_html, encoding='utf-8')
print(f'Generated: {OUT_FILE}  ({len(final_html)//1024}KB)')
