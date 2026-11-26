function initGeometry() {
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

  render();
}

function initRevision() {
  const releases=[[5,102.4],[12,100.9],[23,101.6],[30,101.2]];
  const cutoff=document.getElementById('cutoff');
  function asOf(){
    const day=Number(cutoff.value);
    document.getElementById('day-line').style.left = `${100 * day / Number(cutoff.max)}%`;
    const available=releases.filter(r=>r[0]<=day);
    const current=available.at(-1);
    for (const mark of document.querySelectorAll('.release-mark')) {
      mark.classList.toggle('future', Number(mark.dataset.day) > day);
      mark.classList.toggle('winner', Number(mark.dataset.day) === current?.[0]);
    }
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
  asOf();
}

const theme = document.getElementById('theme');
const menu = document.getElementById('course-menu');
menu.hidden = false;
menu.addEventListener('click', () => {
  menu.setAttribute('aria-expanded', menu.getAttribute('aria-expanded') === 'false');
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

const percent = fraction => (fraction * 100).toFixed(1) + '%';

function initBlend(gap) {
  function render() {
    const delta = Number(gap.value);
    const first = 1 / (1 + Math.exp(-delta));
    const read = 10 * first + 20 * (1 - first);
    document.getElementById('score-gap-value').textContent = delta.toFixed(2);
    document.getElementById('weight-first').style.width = percent(first);
    document.getElementById('blend-marker').style.left = percent((read - 10) / 10);
    document.getElementById('blend-result').textContent =
      `Weights ${percent(first)} and ${percent(1 - first)}. Read ${read.toFixed(2)}°.`;
  }
  gap.addEventListener('input', render);
  render();
}

function initSupports(context) {
  function render() {
    const delta = Number(context.value);
    const query = 1 + delta;
    const firstScore = -(query ** 2) / 2;
    const secondScore = -((query - 4) ** 2) / 2;
    const first = 1 / (1 + Math.exp(secondScore - firstScore));
    const output = query - first + (1 - first);
    document.getElementById('context-value').textContent = delta.toFixed(2);
    document.getElementById('support-marker').style.left = percent(query / 4);
    document.getElementById('support-first').style.width = percent(first);
    document.getElementById('support-result').textContent =
      `Query ${query.toFixed(2)}. Weights ${percent(first)} and ${percent(1 - first)}. Output ${output.toFixed(3)}.`;
  }
  context.addEventListener('input', render);
  render();
}

function initCohorts(select) {
  const models = ['Context-aware pair scorer', 'Learned sequential', 'Learned single-shot top two'];
  // SUPPORT_LINKS.md, evidence recovery counts for the three displayed heads.
  const cohorts = {
    all: { n: 768, hits: [346, 136, 117], note: '468 of 768 pools contain both supports, giving a 60.9% coverage ceiling.' },
    linked: { n: 368, hits: [323, 108, 94], note: 'Both supports are available and one article mentions the other’s title.' },
    unlinked: { n: 100, hits: [23, 28, 23], note: 'Sequential retrieval has the largest count in this smaller cohort.' },
    missing: { n: 300, hits: [0, 0, 0], note: 'Each selector must choose from its pool. A missing support makes success impossible.' },
  };
  const bars = document.getElementById('cohort-bars');
  const rows = models.map(() => {
    const row = document.createElement('div');
    row.className = 'cohort-row';
    row.innerHTML = '<p></p><div class="track" aria-hidden="true"><div class="fill"></div></div>';
    return row;
  });
  bars.replaceChildren(...rows);
  function render() {
    const { n, hits, note } = cohorts[select.value];
    rows.forEach((row, i) => {
      const rate = percent(hits[i] / n);
      row.querySelector('p').textContent = `${models[i]}: ${hits[i]} / ${n} (${rate})`;
      row.querySelector('.fill').style.width = rate;
    });
    document.getElementById('cohort-note').textContent = note;
  }
  select.addEventListener('change', render);
  render();
}

if (document.getElementById('geometry')) initGeometry();
if (document.getElementById('cutoff')) initRevision();
const gap = document.getElementById('score-gap');
if (gap) initBlend(gap);
const context = document.getElementById('context-delta');
if (context) initSupports(context);
const cohort = document.getElementById('cohort');
if (cohort) initCohorts(cohort);

if (document.getElementById('outline')) {
  const oldLinks = {
    '#attention': 'attention.html',
    '#radial': 'radial.html',
    '#geometry-lab': 'radial.html#geometry-lab',
    '#explainer': 'radial.html#explainer',
    '#supports': 'supports.html',
    '#revision': 'revision.html',
    '#revision-lab': 'revision.html#revision-lab',
    '#evidence': 'evidence.html',
  };
  const chapter = oldLinks[location.hash];
  if (chapter) location.replace(chapter);
}
