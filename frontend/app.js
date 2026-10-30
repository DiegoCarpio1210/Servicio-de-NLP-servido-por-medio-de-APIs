'use strict';
const $ = id => document.getElementById(id);
const length = text => Array.from(text).length; // Python offsets/limits count Unicode code points.
let busy = false, turns = [], info = null, evaluation = null, connected = false;
const error = message => { $('error').textContent = message; $('error').hidden = !message; };
function count() {
  for (const [id, max, label] of [['context', 5000, ' characters'], ['question', 500, '']]) {
    const n = length($(id).value), target = $(id+'-count');
    target.textContent = `${n.toLocaleString('en-US')} / ${max.toLocaleString('en-US')}${label}`;
    target.classList.toggle('over-limit', n > max);
  }
}
async function request(path, body) {
  const options = body ? {method:'POST', headers:{'Content-Type':'application/json','X-Pimientero-UI':'1'}, body:JSON.stringify(body)} : {};
  const r = await fetch(path, {...options, signal:AbortSignal.timeout(35000)});
  const data = await r.json();
  if (!r.ok) throw new Error(typeof data.error === 'string' ? data.error : 'Invalid request. Check the input length limits.');
  return data;
}
function element(tag, cls, text) {
  const el = document.createElement(tag); if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text; return el;
}
function answerCard(turn) {
  const result=turn.result;
  const card=element('article','message answer-message');card.append(element('div','answer-label','PIMIENTERO AI'));
  card.append(element('p','answer-text',result.answer===null?'No answer found in the provided context.':result.answer));
  const meta=element('div','answer-meta');
  const score=element('span','',`Score (not a probability): ${result.score.toFixed(3)}`);score.title='Transformed distance from the threshold; it can be high for an incorrect answer.';meta.append(score);
  if(result.start!==null)meta.append(element('span','',`Start offset ${result.start}`));
  meta.append(element('span','',`${(turn.ms/1000).toFixed(2)} s`));
  if(result.answer!==null){const copy=element('button','copy-button','Copy answer');copy.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(result.answer);copy.textContent='Copied';setTimeout(()=>copy.textContent='Copy answer',1600);}catch{error('Copy failed. Select the text and copy it manually.');}});meta.append(copy);}
  card.append(meta);return card;
}
function showView(stats) {
  $('chat-view').hidden = stats; $('stats-view').hidden = !stats;
  $('nav-stats').classList.toggle('active', stats); $('nav-chat').classList.toggle('active', !stats);
  $('page-label').textContent = stats ? 'Model statistics' : 'Questions and answers';
  document.body.classList.remove('menu-open'); $('menu-button').setAttribute('aria-expanded','false');
  if (stats) renderStats();
}
function renderHistory() {
  $('history-count').textContent = turns.length;
  $('history').replaceChildren();
  if (!turns.length) $('history').append(element('p','empty-history','Your questions will appear here.'));
  for (const turn of [...turns].reverse()) {
    const button = element('button','',turn.question); button.title = turn.question;
    button.addEventListener('click', () => {
      if (busy) return;
      showView(false); $('context').value=turn.context; $('question').value=turn.question; count();
      if (!document.getElementById(turn.id)) {
        const user=element('div','message user-message',turn.question);user.id=turn.id;$('messages').append(user);
        if(turn.result)$('messages').append(answerCard(turn));
        else $('messages').append(element('div','error','This request received no response. You can send it again.'));
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
    ['Exact Match',evaluation ? evaluation.metrics.EM.toFixed(2)+' %' : '—',evaluation?.label || 'Evaluation unavailable'],
    ['F1 Score',evaluation ? evaluation.metrics.F1.toFixed(2)+' %' : '—',evaluation?.label || 'Evaluation unavailable'],
    ['Answerable EM',evaluation ? evaluation.answerable.EM.toFixed(2)+' %' : '—','quality40k · 512 answerable questions'],
    ['Unanswerable EM',evaluation ? evaluation.unanswerable.EM.toFixed(2)+' %' : '—','quality40k · 512 unanswerable questions'],
    ['75/25 weighted EM',evaluation ? evaluation.weighted_75_25.EM.toFixed(2)+' %' : '—','quality40k · reference distribution'],
    ['Session requests',String(turns.length),`${total} responses received`],
    ['Latest response',recent ? (recent.ms/1000).toFixed(2)+' s' : '—','Current round-trip time, including the API'],
  ];
  for (const [label,value,note] of items) {
    const card=element('div','metric'); card.append(element('small','',label),element('strong','',value),element('span','',note)); $('metrics-grid').append(card);
  }
  $('model-details').replaceChildren();
  const details = [['Status',connected?'API connected':'API unavailable'],['Model',info?.model || '—'],['Parameters',info ? info.params.toLocaleString('en-US') : '—'],['Checkpoint',info ? `${info.training_steps===2000 ? 'quality40k' : 'continued20k'} · checkpoint-${info.training_steps}` : '—'],['Checkpoint / run steps',info ? `${info.training_steps} / ${info.training_run_steps}` : '—'],['Device','CPU'],['Window / overlap',info ? `${info.max_seq_length} / ${info.doc_stride} tokens` : '—']];
  for (const [key,value] of details) { const row=element('div'); row.append(element('dt','',key),element('dd','',value)); $('model-details').append(row); }
  $('evaluation-source').textContent=evaluation ? 'Source: '+evaluation.source : 'Historical evaluation unavailable.';
}
async function refresh() {
  try { await request('/ui/status'); connected=true;
    info = await request('/ui/info');
  } catch { connected=false; }
  $('status-dot').classList.toggle('online',connected);
  $('connection-text').textContent=connected?'API connected · local':'API unavailable';
  if (!$('stats-view').hidden) renderStats();
}
$('context').addEventListener('input',count); $('question').addEventListener('input',count);
$('question').addEventListener('keydown', e=>{ if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('question-form').requestSubmit();} });
$('question-form').addEventListener('submit',async e=>{
  e.preventDefault(); if(busy)return; error('');
  const context=$('context').value, question=$('question').value;
  if(!context.trim()){error('Add a context before asking a question.');$('context').focus();return;}
  if(!question.trim()){error('Enter a question about your context.');$('question').focus();return;}
  if(length(context)>5000||length(question)>500){error('Context accepts up to 5,000 characters and questions up to 500. Your text has not been truncated.');return;}
  busy=true; $('send').disabled=true; $('clear-chat').disabled=true; $('new-chat').disabled=true;
  const turn={id:crypto.randomUUID(),context,question}; turns.push(turn); renderHistory();
  const user=element('div','message user-message',question); user.id=turn.id; $('messages').append(user);
  const loading=element('div','processing','Searching your context for an answer…'); $('messages').append(loading);
  $('question').value='';count();loading.scrollIntoView({block:'nearest',behavior:'smooth'});
  const start=performance.now();
  try {
    const result=await request('/ui/predict',{id:turn.id,context,question});turn.ms=performance.now()-start;turn.result=result;
    const card=answerCard(turn);loading.replaceWith(card);card.scrollIntoView({block:'nearest',behavior:'smooth'});
  } catch(exc){turn.error=true;loading.replaceWith(element('div','error','This request could not be completed.'));error(exc.name==='TimeoutError'?'The API took too long to respond. Please try again.':exc instanceof TypeError?'Could not connect to the local application. Run INICIAR_PIMIENTERO.bat and try again.':exc.message);$('question').value=question;count();}
  finally {busy=false;$('send').disabled=false;$('clear-chat').disabled=false;$('new-chat').disabled=false;refresh();}
});
function clearConversation(clearContext=false){if(busy)return;$('messages').replaceChildren();$('question').value='';if(clearContext)$('context').value='';error('');count();renderHistory();showView(false);$('context').focus();}
$('new-chat').onclick=()=>clearConversation(true);$('clear-chat').onclick=()=>clearConversation();
$('clear-context').onclick=()=>{$('context').value='';count();$('context').focus();};
$('nav-chat').onclick=()=>showView(false);$('nav-stats').onclick=()=>showView(true);$('model-button').onclick=()=>showView(true);
$('theme-button').onclick=()=>{const dark=document.body.classList.toggle('dark');$('theme-button').setAttribute('aria-pressed',String(dark));$('theme-label').textContent=dark?'Switch to light mode':'Switch to dark mode';};
$('menu-button').onclick=()=>{const open=document.body.classList.toggle('menu-open');$('menu-button').setAttribute('aria-expanded',String(open));};
document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.body.classList.remove('menu-open');$('menu-button').setAttribute('aria-expanded','false');}if(e.key.toLowerCase()==='n'&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName)&&!e.ctrlKey&&!e.metaKey)clearConversation(true);});
request('/ui/evaluation').then(data=>{evaluation=data;if(!$('stats-view').hidden)renderStats();}).catch(()=>{});
refresh();setInterval(refresh,15000);count();
