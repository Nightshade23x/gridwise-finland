"use strict";

const S = [
  {key:"actual_mw", label:"Measured demand", color:"#80c8ff"},
  {key:"model_mw", label:"LightGBM prediction", color:"#59dfba"},
  {key:"selected_baseline_mw", label:"Selected baseline", color:"#ffad66"}
];
const ns = "http://www.w3.org/2000/svg";
const $ = id => document.getElementById(id);

async function loadJSON(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
  return body;
}
function notice(message) { $("notice").textContent = message; $("notice").hidden = !message; }
function fmt(n, digits=1) { return typeof n === "number" ? n.toLocaleString("en-US",{maximumFractionDigits:digits}) : "—"; }

let recentLatestUTC = null;
function recentDateRange() {
  const end = recentLatestUTC ? new Date(recentLatestUTC) : new Date();
  const exclusive = new Date(Date.UTC(end.getUTCFullYear(),end.getUTCMonth(),end.getUTCDate()+1));
  const start = new Date(exclusive.getTime()-7*86400000);
  return [start.toISOString().slice(0,10),exclusive.toISOString().slice(0,10)];
}
async function updateRecentStatus() {
  try {
    const status = await loadJSON("/api/refresh-status");
    recentLatestUTC = status.latest_measurement_utc;
    const format = value => value ? new Date(value).toISOString().replace("T"," ").slice(0,16)+" UTC" : "—";
    $("recent-latest").textContent = format(status.latest_measurement_utc) +
      (typeof status.latest_demand_mw === "number" ? ` · ${fmt(status.latest_demand_mw)} MW` : "");
    $("recent-refresh").textContent = format(status.last_success_utc);
    $("recent-state").textContent = status.status === "ok" ? "Last refresh succeeded" :
      status.status === "failed" ? "Last refresh failed" : "Not configured";
    $("recent-note").textContent = status.status === "ok"
      ? `${fmt(status.stored_rows,0)} measured intervals stored. This does not represent a prediction or verified real-time feed.`
      : status.status === "failed" ? "The last download failed; previously stored measurements remain available. Check the refresh command output."
      : "Run the standalone data-refresh command to enable recent measurements.";
  } catch(error) {
    $("recent-state").textContent = "Status unavailable";
    $("recent-note").textContent = "Unable to read refresh status.";
  }
}
async function updateSummary() {
  try {
    const data = await loadJSON("/api/summary");
    const metrics = data.evaluation.metrics;
    $("model-mae").innerHTML = `${fmt(metrics.lightgbm?.mae_mw)} <em>MW</em>`;
    $("baseline-mae").innerHTML = `${fmt(metrics.selected_baseline?.mae_mw)} <em>MW</em>`;
    const base = metrics.selected_baseline?.mae_mw;
    const model = metrics.lightgbm?.mae_mw;
    $("improvement").textContent = base > 0 && typeof model === "number" ? `${fmt(100*(base-model)/base)}%` : "—";
    $("coverage").textContent = `${fmt(data.history.completeness_percent,2)}%`;
  } catch (error) { notice(error.message); }
}

function svgElement(tag, attrs = {}, text = "") {
  const e = document.createElementNS(ns, tag);
  for (const [key,value] of Object.entries(attrs)) e.setAttribute(key,String(value));
  if (text) e.textContent = text;
  return e;
}

/** Render distinct UTC ticks for short or long chart windows. */
function axisTickLabel(timestamp, spanMs, interval) {
  const iso = new Date(timestamp).toISOString();
  if (interval === "day" || spanMs >= 6 * 86400000) return iso.slice(0, 10);
  if (spanMs <= 36 * 3600000) return iso.slice(11, 16);
  return `${iso.slice(5, 10)} ${iso.slice(11, 16)}`;
}

function drawChart(points, kind, interval="hour") {
  const svg = $("chart"); svg.replaceChildren();
  const legend = $("legend"); legend.replaceChildren();
  const cols = kind === "test" ? S : [{key:"demand_mw",label:"Measured demand",color:"#80c8ff"}];
  for (const series of cols) {
    const node = document.createElement("span"); node.className = "legend-item";
    const swatch = document.createElement("span"); swatch.className = "swatch"; swatch.style.background = series.color;
    node.append(swatch,document.createTextNode(series.label)); legend.append(node);
  }
  const W=960,H=380,left=73,right=28,top=23,bottom=47,plotW=W-left-right,plotH=H-top-bottom;
  if (!points.length) {
    svg.append(svgElement("text",{x:W/2,y:H/2,"text-anchor":"middle",fill:"#9aafc1"},"No observations in the selected period"));
    return;
  }
  const xValues=points.map(p=>Date.parse(p.timestamp_utc));
  const vals=points.flatMap(p=>cols.map(s=>p[s.key]).filter(v=>typeof v==="number"));
  if (!vals.length) {
    svg.append(svgElement("text",{x:W/2,y:H/2,"text-anchor":"middle",fill:"#9aafc1"},"No measured values in selected period"));
    return;
  }
  const minValue=Math.min(...vals),maxValue=Math.max(...vals);
  const padding=Math.max((maxValue-minValue)*.12,150),yMin=Math.max(0,minValue-padding),yMax=maxValue+padding;
  const xMin=xValues[0],xMax=Math.max(xValues[xValues.length-1],xMin+1);
  const X=t=>left+(t-xMin)/(xMax-xMin)*plotW;
  const Y=n=>top+(yMax-n)/(yMax-yMin)*plotH;
  for (let i=0;i<=5;i++) {
    const y=top+i*plotH/5;
    svg.append(svgElement("line",{x1:left,x2:W-right,y1:y,y2:y,class:"chart-grid"}));
    svg.append(svgElement("text",{x:left-12,y:y+4,"text-anchor":"end",class:"chart-axis"},Math.round(yMax-i*(yMax-yMin)/5).toLocaleString()));
  }
  // Tick positions are actual observations. This avoids duplicate date labels
  // for daily charts and uses hour labels for a single-day 15-minute view.
  const tickCount=Math.min(7, points.length);
  for(let i=0;i<tickCount;i++) {
    const idx=tickCount===1?0:Math.round(i*(points.length-1)/(tickCount-1));
    const t=xValues[idx];
    const stamp=axisTickLabel(t,xMax-xMin,interval);
    svg.append(svgElement("text",{x:X(t),y:H-15,"text-anchor":"middle",class:"chart-axis"},stamp));
  }
  for(const series of cols) {
    // Separate missing-data segments instead of drawing misleading connecting lines.
    let segments=[],current=[];
    points.forEach((p,i)=>{
      const v=p[series.key];
      if(typeof v==="number") current.push(`${X(xValues[i]).toFixed(2)},${Y(v).toFixed(2)}`);
      else if(current.length){segments.push(current);current=[];}
    });
    if(current.length)segments.push(current);
    segments.forEach(seg=>{
      if(seg.length===1) {
        const [cx,cy]=seg[0].split(",");
        svg.append(svgElement("circle",{cx,cy,r:2.5,fill:series.color}));
      } else svg.append(svgElement("polyline",{points:seg.join(" "),stroke:series.color,class:"chart-line"}));
    });
  }
  // Full-width overlay supports mouse readout without thousands of circles.
  const overlay=svgElement("rect",{x:left,y:top,width:plotW,height:plotH,class:"chart-overlay"});
  const crosshair=svgElement("line",{x1:left,x2:left,y1:top,y2:H-bottom,stroke:"#9bb9c8","stroke-dasharray":"4 5",visibility:"hidden"});
  svg.append(crosshair,overlay);
  const tooltip=$("tooltip");
  overlay.addEventListener("mousemove",event=>{
    const rect=svg.getBoundingClientRect();
    const rx=(event.clientX-rect.left)/rect.width*W;
    const target=xMin+(Math.max(left,Math.min(W-right,rx))-left)/plotW*(xMax-xMin);
    let lo=0,hi=xValues.length-1;
    while(lo<hi){const mid=(lo+hi)>>1;if(xValues[mid]<target)lo=mid+1;else hi=mid;}
    const idx=lo>0&&Math.abs(xValues[lo-1]-target)<Math.abs(xValues[lo]-target)?lo-1:lo;
    const p=points[idx],x=X(xValues[idx]);
    crosshair.setAttribute("x1",String(x));crosshair.setAttribute("x2",String(x));crosshair.setAttribute("visibility","visible");
    tooltip.replaceChildren();
    const first=document.createElement("strong");
    const stamp=new Date(p.timestamp_utc).toISOString();
    first.textContent=interval==="day"?`${stamp.slice(0,10)} (UTC day)`:`${stamp.replace("T"," ").slice(0,16)} UTC`;
    tooltip.append(first);
    cols.forEach(s=>{const line=document.createElement("div");line.textContent=`${s.label}: ${fmt(p[s.key])} MW`;line.style.color=s.color;tooltip.append(line);});
    tooltip.hidden=false;
    const px=(x/W)*rect.width;
    tooltip.style.left=`${Math.min(rect.width-205,Math.max(12,px+12))}px`;
    tooltip.style.top=`${Math.max(12,(event.clientY-rect.top)-tooltip.offsetHeight-16)}px`;
  });
  overlay.addEventListener("mouseleave",()=>{tooltip.hidden=true;crosshair.setAttribute("visibility","hidden");});
}

function showInsights(data) {
  $("insights-headline").textContent = data.headline;
  const metrics = $("insights-metrics"); metrics.replaceChildren();
  for (const metric of data.metrics) {
    const card = document.createElement("div"); card.className = "insight-metric";
    const label = document.createElement("span"); label.textContent = metric.label;
    const value = document.createElement("strong"); value.textContent = metric.value;
    card.append(label, value); metrics.append(card);
  }
  const details = $("insights-details"); details.replaceChildren();
  for (const item of data.details) {
    const li = document.createElement("li"); li.textContent = item; details.append(li);
  }
  $("insights-implication").textContent = data.implication;
}

// Export precisely the points and resolution currently displayed, including
// missing values as blank cells. The downloaded file is never sent elsewhere.
let currentSeries = null;
function exportChartCSV() {
  if (!currentSeries || !currentSeries.points.length) return;
  const columns=currentSeries.kind === "test" ? ["timestamp_utc", "actual_mw", "model_mw", "selected_baseline_mw"] : ["timestamp_utc", "demand_mw"];
  const escapeCSV=value=>value===null||value===undefined?"":`"${String(value).replaceAll('"','""')}"`;
  const csv=[columns.join(","), ...currentSeries.points.map(p=>columns.map(k=>escapeCSV(p[k])).join(","))].join("\r\n")+"\r\n";
  const blob=new Blob([csv],{type:"text/csv;charset=utf-8"});
  const url=URL.createObjectURL(blob);
  const link=document.createElement("a");
  link.href=url;
  link.download=`gridwise_${currentSeries.kind}_${currentSeries.startDate}_${currentSeries.endDate}_${currentSeries.interval}.csv`;
  document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$("download-csv").addEventListener("click",exportChartCSV);

let chartRequestSequence = 0;
async function updateChart(event) {
  event?.preventDefault();
  const requestId=++chartRequestSequence;
  notice("");
  const button=$("apply");button.disabled=true;button.textContent="Loading…";
  $("download-csv").disabled=true;
  $("insights-headline").textContent="Calculating insights for your selection…";
  try {
    const params=new URLSearchParams({kind:$("kind").value,start:$("start").value,end:$("end").value,interval:$("interval").value});
    const insightParams=new URLSearchParams({kind:$("kind").value,start:$("start").value,end:$("end").value});
    const [data, insights]=await Promise.all([
      loadJSON(`/api/series?${params}`),
      loadJSON(`/api/insights?${insightParams}`)
    ]);
    if(requestId!==chartRequestSequence)return;
    drawChart(data.points,data.kind,data.interval);
    currentSeries={...data,startDate:params.get("start"),endDate:params.get("end")};
    $("download-csv").disabled=!data.points.length;
    showInsights(insights);
    $("point-count").textContent=`${data.count.toLocaleString()} ${data.interval} observations shown`;
  } catch(error) {
    if(requestId===chartRequestSequence){
      notice(error.message);
      currentSeries=null;
      $("download-csv").disabled=true;
      $("insights-headline").textContent="Unable to calculate insights for this selection.";
      $("insights-metrics").replaceChildren();
      $("insights-details").replaceChildren();
      $("insights-implication").textContent="";
    }
  } finally {
    if(requestId===chartRequestSequence){button.disabled=false;button.textContent="Apply";}
  }
}

$("filters").addEventListener("submit",updateChart);
$("kind").addEventListener("change",()=>{
  if($("kind").value === "history") {$("start").value="2024-01-01";$("end").value="2024-01-08";}
  else if($("kind").value === "recent") {[$("start").value,$("end").value] = recentDateRange();}
  else {$("start").value="2026-07-01";$("end").value="2026-07-08";}
  updateChart();
});
$("show-recent").addEventListener("click",()=>{
  $("kind").value="recent";
  $("kind").dispatchEvent(new Event("change"));
  $("filters").scrollIntoView({behavior:"smooth",block:"center"});
});
updateSummary();updateRecentStatus();updateChart();
