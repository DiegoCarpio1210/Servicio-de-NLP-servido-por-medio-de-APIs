'use strict';
const $ = id => document.getElementById(id);
const length = text => Array.from(text).length; // Python offsets/limits count Unicode code points.
let busy = false, turns = [], info = null, evaluation = null, connected = false;
const error = message => { $('error').textContent = message; $('error').hidden = !message; };
function count() {
  for (const [id, max, label] of [['context', 5000, ' caracteres'], ['question', 500, '']]) {
    const n = length($(id).value), target = $(id+'-count');
    target.textContent = `${n.toLocaleString('es')} / ${max.toLocaleString('es')}${label}`;
    target.classList.toggle('over-limit', n > max);
  }
}
async function request(path, body) {
  const options = body ? {method:'POST', headers:{'Content-Type':'application/json','X-Pimientero-UI':'1'}, body:JSON.stringify(body)} : {};
  const r = await fetch(path, {...options, signal:AbortSignal.timeout(35000)});
  const data = await r.json();
  if (!r.ok) throw new Error(typeof data.error === 'string' ? data.error : 'La solicitud no es válida. Revisa los límites del texto.');
  return data;
}
function element(tag, cls, text) {
  const el = document.createElement(tag); if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text; return el;
}
function answerCard(turn) {
  const result=turn.result;
  const card=element('article','message answer-message');card.append(element('div','answer-label','PIMIENTERO AI'));
  card.append(element('p','answer-text',result.answer===null?'No encuentro una respuesta en el contexto proporcionado.':result.answer));
  const meta=element('div','answer-meta');
  const score=element('span','',`Score (no probabilidad): ${result.score.toFixed(3)}`);score.title='Distancia transformada al umbral; puede ser alta en una respuesta incorrecta.';meta.append(score);
  if(result.start!==null)meta.append(element('span','',`Posición inicial ${result.start}`));
  meta.append(element('span','',`${(turn.ms/1000).toFixed(2)} s`));
  if(result.answer!==null){const copy=element('button','copy-button','Copiar respuesta');copy.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(result.answer);copy.textContent='Copiado';setTimeout(()=>copy.textContent='Copiar respuesta',1600);}catch{error('No se pudo copiar. Selecciona el texto y cópialo manualmente.');}});meta.append(copy);}
  card.append(meta);return card;
}
function showView(stats) {
  $('chat-view').hidden = stats; $('stats-view').hidden = !stats;
  $('nav-stats').classList.toggle('active', stats); $('nav-chat').classList.toggle('active', !stats);
  $('page-label').textContent = stats ? 'Estadísticas del modelo' : 'Preguntas y respuestas';
  document.body.classList.remove('menu-open'); $('menu-button').setAttribute('aria-expanded','false');
  if (stats) renderStats();
}
function renderHistory() {
  $('history-count').textContent = turns.length;
  $('history').replaceChildren();
  if (!turns.length) $('history').append(element('p','empty-history','Tus preguntas aparecerán aquí.'));
  for (const turn of [...turns].reverse()) {
    const button = element('button','',turn.question); button.title = turn.question;
    button.addEventListener('click', () => {
      if (busy) return;
      showView(false); $('context').value=turn.context; $('question').value=turn.question; count();
      if (!document.getElementById(turn.id)) {
        const user=element('div','message user-message',turn.question);user.id=turn.id;$('messages').append(user);
        if(turn.result)$('messages').append(answerCard(turn));
        else $('messages').append(element('div','error','Esta consulta no recibió una respuesta. Puedes volver a enviarla.'));
      }
      document.getElementById(turn.id)?.scrollIntoView({behavior:'smooth',block:'center'});
    }); $('history').append(button);
  }
}
function renderStats() {
  $('metrics-grid').replaceChildren();
  const total = turns.filter(t=>t.result).length;
  const recent = [...turns].reverse().find(t=>t.result);
  const items = [
    ['Exact Match',evaluation ? evaluation.metrics.EM.toFixed(2)+' %' : '—',evaluation?.label || 'Evaluación no disponible'],
    ['F1 Score',evaluation ? evaluation.metrics.F1.toFixed(2)+' %' : '—',evaluation?.label || 'Evaluación no disponible'],
    ['EM respondibles',evaluation ? evaluation.answerable.EM.toFixed(2)+' %' : '—','quality40k · 512 preguntas respondibles'],
    ['EM imposibles',evaluation ? evaluation.unanswerable.EM.toFixed(2)+' %' : '—','quality40k · 512 preguntas imposibles'],
    ['EM ponderado 75/25',evaluation ? evaluation.weighted_75_25.EM.toFixed(2)+' %' : '—','quality40k · distribución de referencia'],
    ['Consultas de esta sesión',String(turns.length),`${total} respuestas recibidas`],
    ['Última respuesta',recent ? (recent.ms/1000).toFixed(2)+' s' : '—','Tiempo actual de ida y vuelta; incluye la API'],
  ];
  for (const [label,value,note] of items) {
    const card=element('div','metric'); card.append(element('small','',label),element('strong','',value),element('span','',note)); $('metrics-grid').append(card);
  }
  $('model-details').replaceChildren();
  const details = [['Estado',connected?'API conectada':'API no disponible'],['Modelo',info?.model || '—'],['Parámetros',info ? info.params.toLocaleString('es') : '—'],['Checkpoint',info ? `${info.training_steps===2000 ? 'quality40k' : 'continued20k'} · checkpoint-${info.training_steps}` : '—'],['Pasos del checkpoint / ejecución',info ? `${info.training_steps} / ${info.training_run_steps}` : '—'],['Dispositivo','CPU'],['Ventana / solapamiento',info ? `${info.max_seq_length} / ${info.doc_stride} tokens` : '—']];
  for (const [key,value] of details) { const row=element('div'); row.append(element('dt','',key),element('dd','',value)); $('model-details').append(row); }
  $('evaluation-source').textContent=evaluation ? 'Fuente: '+evaluation.source : 'Evaluación histórica no disponible.';
}
async function refresh() {
  try { await request('/ui/status'); connected=true;
    info = await request('/ui/info');
  } catch { connected=false; }
  $('status-dot').classList.toggle('online',connected);
  $('connection-text').textContent=connected?'API conectada · local':'API no disponible';
  if (!$('stats-view').hidden) renderStats();
}
$('context').addEventListener('input',count); $('question').addEventListener('input',count);
$('question').addEventListener('keydown', e=>{ if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('question-form').requestSubmit();} });
$('question-form').addEventListener('submit',async e=>{
  e.preventDefault(); if(busy)return; error('');
  const context=$('context').value, question=$('question').value;
  if(!context.trim()){error('Añade un contexto antes de preguntar.');$('context').focus();return;}
  if(!question.trim()){error('Escribe una pregunta sobre tu contexto.');$('question').focus();return;}
  if(length(context)>5000||length(question)>500){error('El contexto admite hasta 5.000 caracteres y la pregunta hasta 500. El texto no se ha recortado.');return;}
  busy=true; $('send').disabled=true; $('clear-chat').disabled=true; $('new-chat').disabled=true;
  const turn={id:crypto.randomUUID(),context,question}; turns.push(turn); renderHistory();
  const user=element('div','message user-message',question); user.id=turn.id; $('messages').append(user);
  const loading=element('div','processing','Buscando la respuesta en tu contexto…'); $('messages').append(loading);
  $('question').value='';count();loading.scrollIntoView({block:'nearest',behavior:'smooth'});
  const start=performance.now();
  try {
    const result=await request('/ui/predict',{id:turn.id,context,question});turn.ms=performance.now()-start;turn.result=result;
    const card=answerCard(turn);loading.replaceWith(card);card.scrollIntoView({block:'nearest',behavior:'smooth'});
  } catch(exc){turn.error=true;loading.replaceWith(element('div','error','No se pudo completar esta consulta.'));error(exc.name==='TimeoutError'?'La API tardó demasiado en responder. Inténtalo de nuevo.':exc instanceof TypeError?'No se pudo conectar con la aplicación local. Inicia INICIAR_PIMIENTERO.bat e inténtalo de nuevo.':exc.message);$('question').value=question;count();}
  finally {busy=false;$('send').disabled=false;$('clear-chat').disabled=false;$('new-chat').disabled=false;refresh();}
});
function clearConversation(clearContext=false){if(busy)return;$('messages').replaceChildren();$('question').value='';if(clearContext)$('context').value='';error('');count();renderHistory();showView(false);$('context').focus();}
$('new-chat').onclick=()=>clearConversation(true);$('clear-chat').onclick=()=>clearConversation();
$('clear-context').onclick=()=>{$('context').value='';count();$('context').focus();};
$('nav-chat').onclick=()=>showView(false);$('nav-stats').onclick=()=>showView(true);$('model-button').onclick=()=>showView(true);
$('theme-button').onclick=()=>{const dark=document.body.classList.toggle('dark');$('theme-button').setAttribute('aria-pressed',String(dark));$('theme-label').textContent=dark?'Cambiar a modo claro':'Cambiar a modo oscuro';};
$('menu-button').onclick=()=>{const open=document.body.classList.toggle('menu-open');$('menu-button').setAttribute('aria-expanded',String(open));};
document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.body.classList.remove('menu-open');$('menu-button').setAttribute('aria-expanded','false');}if(e.key.toLowerCase()==='n'&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName)&&!e.ctrlKey&&!e.metaKey)clearConversation(true);});
request('/ui/evaluation').then(data=>{evaluation=data;if(!$('stats-view').hidden)renderStats();}).catch(()=>{});
refresh();setInterval(refresh,15000);count();
