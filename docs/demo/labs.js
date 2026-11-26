const keys=[{x:-1.55,y:.85,c:'#ffb19a'},{x:-.75,y:-1.05,c:'#fed78c'},{x:.95,y:.8,c:'#ada8fb'},{x:1.65,y:-.47,c:'#87beee'}];
const ids=['qx','qy','sigma','metric'];
const sliders=Object.fromEntries(ids.map(id=>[id,document.getElementById(id)]));
const canvas=document.getElementById('geometry');
const ctx=canvas.getContext('2d');
const scale=124, cx=360, cy=245;
const sx=x=>cx+x*scale, sy=y=>cy-y*scale;
const weights=document.getElementById('weights');
const bars=keys.map((key,i)=>{
  const row=document.createElement('div');
  row.className='weight';
  row.innerHTML='<span>Key '+(i+1)+'</span><div class="track"><div class="fill"></div></div><span class="number"></span>';
  row.querySelector('.fill').style.background=key.c;
  weights.append(row);
  return row;
});
function render(){
  const [x,y,sigma,metric]=ids.map(id=>Number(sliders[id].value));
  for(const id of ids) document.getElementById(id+'-value').textContent=Number(sliders[id].value).toFixed(2);
  const scores=keys.map(k=>-(metric*(x-k.x)**2+(y-k.y)**2)/(2*sigma**2));
  const max=Math.max(...scores), shifted=scores.map(s=>Math.exp(s-max));
  const p=shifted.map(a=>a/shifted.reduce((sum,b)=>sum+b,0));
  ctx.fillStyle='#0e192c';ctx.fillRect(0,0,720,490);
  ctx.lineWidth=1;ctx.strokeStyle='#26364e';ctx.beginPath();
  for(let i=-3;i<=3;i+=.5){ctx.moveTo(sx(i),0);ctx.lineTo(sx(i),490);}
  for(let i=-2;i<=2;i+=.5){ctx.moveTo(0,sy(i));ctx.lineTo(720,sy(i));}
  ctx.stroke();
  for(const t of [2,1.5,1,.5]){
    ctx.beginPath();ctx.ellipse(sx(x),sy(y),t*sigma*scale/Math.sqrt(metric),t*sigma*scale,0,0,Math.PI*2);
    ctx.fillStyle='rgba(83,224,187,.045)';ctx.fill();
    ctx.strokeStyle='rgba(117,239,208,.47)';ctx.lineWidth=1.5;ctx.stroke();
  }
  keys.forEach((k,i)=>{
    ctx.beginPath();ctx.moveTo(sx(x),sy(y));ctx.lineTo(sx(k.x),sy(k.y));
    ctx.globalAlpha=.14+.81*p[i];ctx.strokeStyle=k.c;ctx.lineWidth=1+6*p[i];ctx.stroke();ctx.globalAlpha=1;
    ctx.beginPath();ctx.arc(sx(k.x),sy(k.y),7+7*p[i],0,2*Math.PI);ctx.fillStyle=k.c;ctx.fill();
    ctx.font='700 19px system-ui';ctx.fillText('K'+(i+1),sx(k.x)+15,sy(k.y)-9);
    bars[i].querySelector('.fill').style.width=(100*p[i])+'%';
    bars[i].querySelector('.number').textContent=(100*p[i]).toFixed(1)+'%';
  });
  ctx.beginPath();ctx.arc(sx(x),sy(y),10,0,2*Math.PI);ctx.fillStyle='#75efd0';ctx.fill();
  ctx.strokeStyle='#0c1425';ctx.lineWidth=3;ctx.stroke();
  ctx.fillStyle='#75efd0';ctx.font='700 20px system-ui';ctx.fillText('q',sx(x)+15,sy(y)-9);
}
function move(event){
  const b=canvas.getBoundingClientRect();
  sliders.qx.value=(((event.clientX-b.left)/b.width*720-cx)/scale).toFixed(2);
  sliders.qy.value=((cy-(event.clientY-b.top)/b.height*490)/scale).toFixed(2);
  render();
}
canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);move(e);});
canvas.addEventListener('pointermove',e=>{if(canvas.hasPointerCapture(e.pointerId)) move(e);});
Object.values(sliders).forEach(s=>s.addEventListener('input',render));
document.getElementById('reset').addEventListener('click',()=>{ids.forEach(id=>{sliders[id].value=sliders[id].defaultValue;});render();});
const releases=[[5,102.4],[12,100.9],[23,101.6],[30,101.2]];
const cutoff=document.getElementById('cutoff');
function asOf(){
  const day=Number(cutoff.value);
  const available=releases.filter(r=>r[0]<=day);
  const current=available.at(-1);
  document.getElementById('cutoff-value').textContent=day;
  document.getElementById('choice').textContent=current?'Selected value '+current[1]:'No record available yet';
  const body=document.getElementById('revisions');
  body.replaceChildren(...releases.map((r,i)=>{
    const tr=document.createElement('tr');
    if(r===current) tr.className='winner';
    [i+1,r[0],r[1],r===current?'Selected':r[0]>day?'Future':'Superseded'].forEach(v=>{
      const td=document.createElement('td');td.textContent=v;tr.append(td);
    });
    return tr;
  }));
}
cutoff.addEventListener('input',asOf);
render();asOf();

const theme = document.getElementById('theme');
const menu = document.getElementById('course-menu');
menu.hidden = false;
menu.addEventListener('click', () => {
  menu.setAttribute('aria-expanded', menu.getAttribute('aria-expanded') === 'false');
});
document.querySelectorAll('#course-navigation a[href^="#"]').forEach(link => {
  link.addEventListener('click', () => {
    if (menu.getClientRects().length) {
      menu.setAttribute('aria-expanded', 'false');
      const chapter = document.querySelector(link.hash);
      chapter.tabIndex = -1;
      chapter.focus({ preventScroll: true });
    }
  });
});

function applyTheme(value) {
  if (value === 'light' || value === 'dark') {
    document.documentElement.dataset.theme = value;
  } else {
    delete document.documentElement.dataset.theme;
  }
}
try {
  const savedTheme = localStorage.getItem('kam-course-theme');
  if (savedTheme === 'light' || savedTheme === 'dark') theme.value = savedTheme;
} catch {
  // The course still follows the system when browser storage is unavailable.
}
applyTheme(theme.value);
theme.addEventListener('change', () => {
  applyTheme(theme.value);
  try {
    localStorage.setItem('kam-course-theme', theme.value);
  } catch {
    // Keep the current selection for this page even without storage.
  }
});
