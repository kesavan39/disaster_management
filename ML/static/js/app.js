/**
 * Disaster Response Command Center - Frontend Logic
 * Connects UI inputs to Stage 01 ML REST API
 */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSimulator();
  loadDistricts();
  loadLeaderboard();
  loadEdaAndEval();
  loadFieldBriefing();
});

// 1. Navigation Tabs
function initTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      
      tab.classList.add('active');
      const targetId = tab.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');
    });
  });
}

// 2. Real-Time Risk Simulator
function initSimulator() {
  const sliders = [
    'water_level_m', 'rainfall_mm_24h', 'emergency_calls_6h',
    'road_closures', 'power_outage_probability', 'shelter_beds_available', 'ambulances_available'
  ];

  sliders.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    const valSpan = document.getElementById(id + '_val');
    
    el.addEventListener('input', (e) => {
      let val = e.target.value;
      if (id === 'power_outage_probability') {
        valSpan.textContent = Math.round(val * 100) + '%';
      } else {
        valSpan.textContent = val;
      }
      triggerPrediction();
    });
  });

  // Initial trigger
  triggerPrediction();
}

async function triggerPrediction() {
  const payload = {
    water_level_m: parseFloat(document.getElementById('water_level_m').value),
    rainfall_mm_24h: parseFloat(document.getElementById('rainfall_mm_24h').value),
    emergency_calls_6h: parseFloat(document.getElementById('emergency_calls_6h').value),
    road_closures: parseFloat(document.getElementById('road_closures').value),
    power_outage_probability: parseFloat(document.getElementById('power_outage_probability').value),
    shelter_beds_available: parseFloat(document.getElementById('shelter_beds_available').value),
    ambulances_available: parseFloat(document.getElementById('ambulances_available').value),
    rainfall_mm_6h: parseFloat(document.getElementById('rainfall_mm_24h').value) * 0.35
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.status === 'success') {
      renderPredictionResult(data);
    }
  } catch (err) {
    console.error("Prediction API error:", err);
  }
}

function renderPredictionResult(data) {
  const scoreEl = document.getElementById('res_score');
  const levelBadge = document.getElementById('res_badge');
  const confSpan = document.getElementById('res_conf');
  const actionText = document.getElementById('res_action');

  scoreEl.textContent = data.risk_score;
  confSpan.textContent = data.confidence_score + '%';
  actionText.textContent = data.tactical_action;

  levelBadge.className = 'risk-badge-large ';
  if (data.risk_level === 'Severe') {
    levelBadge.classList.add('badge-sev');
    levelBadge.textContent = 'CRITICAL SEVERE RISK';
  } else if (data.risk_level === 'Moderate') {
    levelBadge.classList.add('badge-mod');
    levelBadge.textContent = 'MODERATE RISK ADVISORY';
  } else {
    levelBadge.classList.add('badge-low');
    levelBadge.textContent = 'LOW RISK / NORMAL';
  }

  // Update probability progress bars
  const probs = data.probabilities || {};
  document.getElementById('prob_low').style.width = ((probs.Low || 0) * 100) + '%';
  document.getElementById('prob_mod').style.width = ((probs.Moderate || 0) * 100) + '%';
  document.getElementById('prob_sev').style.width = ((probs.Severe || 0) * 100) + '%';

  document.getElementById('val_prob_low').textContent = Math.round((probs.Low || 0) * 100) + '%';
  document.getElementById('val_prob_mod').textContent = Math.round((probs.Moderate || 0) * 100) + '%';
  document.getElementById('val_prob_sev').textContent = Math.round((probs.Severe || 0) * 100) + '%';
}

// 3. Load District Triage Grid
async function loadDistricts() {
  try {
    const res = await fetch('/api/districts');
    const data = await res.json();
    if (!data.districts) return;

    let severeCount = 0;
    let moderateCount = 0;

    const tbody = document.getElementById('districts_tbody');
    tbody.innerHTML = '';

    data.districts.forEach(d => {
      if (d.risk_level === 'Severe') severeCount++;
      if (d.risk_level === 'Moderate') moderateCount++;

      const tr = document.createElement('tr');
      let badgeClass = 'badge-low';
      if (d.risk_level === 'Severe') badgeClass = 'badge-sev';
      if (d.risk_level === 'Moderate') badgeClass = 'badge-mod';

      tr.innerHTML = `
        <td style="font-weight:600; color:#fff;">${d.district}</td>
        <td>${d.river}</td>
        <td style="font-family:var(--font-mono);">${d.water_level_m} m</td>
        <td style="font-family:var(--font-mono);">${d.rainfall_mm_24h} mm</td>
        <td>${d.emergency_calls_6h}</td>
        <td>${d.road_closures}</td>
        <td style="font-family:var(--font-mono); font-weight:700;">${d.risk_score}</td>
        <td><span class="status-badge ${badgeClass}" style="padding:4px 10px; font-size:0.75rem;">${d.risk_level}</span></td>
      `;
      tbody.appendChild(tr);
    });

    document.getElementById('metric_total_districts').textContent = data.districts.length;
    document.getElementById('metric_severe_count').textContent = severeCount;
  } catch (err) {
    console.error("Error loading districts:", err);
  }
}

// 4. Load Feature Importance Leaderboard
async function loadLeaderboard() {
  try {
    const res = await fetch('/api/leaderboard');
    const leaderboard = await res.json();
    if (!Array.isArray(leaderboard)) return;

    const container = document.getElementById('leaderboard_container');
    container.innerHTML = '';

    leaderboard.slice(0, 10).forEach(item => {
      const row = document.createElement('div');
      row.style.marginBottom = '14px';

      row.innerHTML = `
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px; color:var(--text-main);">
          <span><strong>#${item.rank}</strong> ${formatFeatureName(item.feature)}</span>
          <span style="font-family:var(--font-mono); color:var(--accent-cyan);">${item.percentage}%</span>
        </div>
        <div style="height:10px; background:rgba(255,255,255,0.05); border-radius:5px; overflow:hidden;">
          <div style="width:${Math.max(item.percentage, 3)}%; height:100%; background:linear-gradient(to right, var(--accent-blue), var(--accent-cyan)); border-radius:5px;"></div>
        </div>
      `;
      container.appendChild(row);
    });

    // Update top predictor metric card
    if (leaderboard.length > 0) {
      document.getElementById('metric_top_predictor').textContent = formatFeatureName(leaderboard[0].feature);
    }
  } catch (err) {
    console.error("Error loading leaderboard:", err);
  }
}

function formatFeatureName(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// 5. Load EDA & Evaluation Metrics
async function loadEdaAndEval() {
  try {
    const edaRes = await fetch('/api/eda');
    const edaData = await edaRes.json();
    if (edaData.false_alarm_diagnostics) {
      const faRate = edaData.false_alarm_diagnostics.false_alarm_rate;
      document.getElementById('metric_false_alarm_rate').textContent = (faRate * 100).toFixed(1) + '%';
    }

    const evalRes = await fetch('/api/evaluation');
    const evalData = await evalRes.json();
    if (evalData.test_metrics) {
      document.getElementById('metric_model_accuracy').textContent = (evalData.test_metrics.accuracy * 100).toFixed(1) + '%';
      renderEvaluationDetails(evalData.test_metrics);
    }
  } catch (err) {
    console.error("Error loading EDA/Eval metrics:", err);
  }
}

function renderEvaluationDetails(metrics) {
  const container = document.getElementById('eval_details_container');
  if (!container) return;

  container.innerHTML = `
    <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin-bottom:20px; text-align:center;">
      <div style="background:rgba(255,255,255,0.03); padding:16px; border-radius:10px;">
        <div style="font-size:0.8rem; color:var(--text-muted);">ACCURACY</div>
        <div style="font-size:1.8rem; font-weight:700; color:var(--accent-cyan);">${(metrics.accuracy*100).toFixed(1)}%</div>
      </div>
      <div style="background:rgba(255,255,255,0.03); padding:16px; border-radius:10px;">
        <div style="font-size:0.8rem; color:var(--text-muted);">WEIGHTED F1 SCORE</div>
        <div style="font-size:1.8rem; font-weight:700; color:#fff;">${metrics.weighted_f1}</div>
      </div>
      <div style="background:rgba(255,255,255,0.03); padding:16px; border-radius:10px;">
        <div style="font-size:0.8rem; color:var(--text-muted);">OVERCONFIDENCE ERROR</div>
        <div style="font-size:1.8rem; font-weight:700; color:var(--severity-low);">${(metrics.overconfidence_rate*100).toFixed(1)}%</div>
      </div>
    </div>
  `;
}

// 6. Load Field Briefing Sheet
async function loadFieldBriefing() {
  try {
    const res = await fetch('/api/field_briefing');
    const briefing = await res.json();
    if (!briefing.threat_levels) return;

    const paper = document.getElementById('field_briefing_paper');
    paper.innerHTML = `
      <div class="briefing-header">
        <h2>${briefing.title}</h2>
        <p style="color:#4b5563; font-size:0.85rem; font-weight:500;">Target Audience: ${briefing.target_audience} | SEOC Contact: ${briefing.field_contact}</p>
      </div>

      <div class="briefing-section">
        <h3>1. TRIAGE THREAT LEVEL PROTOCOLS</h3>
        <div class="briefing-grid">
          <div class="briefing-box b-green">
            <h4 style="font-weight:700; margin-bottom:6px;">GREEN (LOW RISK)</h4>
            <p><strong>Trigger:</strong> ${briefing.threat_levels.LOW.trigger_condition}</p>
            <p style="margin-top:6px;"><strong>Action:</strong> ${briefing.threat_levels.LOW.tactical_action}</p>
          </div>
          <div class="briefing-box b-yellow">
            <h4 style="font-weight:700; margin-bottom:6px;">AMBER (MODERATE RISK)</h4>
            <p><strong>Trigger:</strong> ${briefing.threat_levels.MODERATE.trigger_condition}</p>
            <p style="margin-top:6px;"><strong>Action:</strong> ${briefing.threat_levels.MODERATE.tactical_action}</p>
          </div>
          <div class="briefing-box b-red">
            <h4 style="font-weight:700; margin-bottom:6px;">RED (SEVERE RISK)</h4>
            <p><strong>Trigger:</strong> ${briefing.threat_levels.SEVERE.trigger_condition}</p>
            <p style="margin-top:6px;"><strong>Action:</strong> ${briefing.threat_levels.SEVERE.tactical_action}</p>
          </div>
        </div>
      </div>

      <div class="briefing-section" style="background:#f3f4f6; padding:12px; border-radius:8px; border-left:4px solid #1e3a8a;">
        <h4 style="font-size:0.9rem; color:#1e3a8a; margin-bottom:4px;">⚠️ FALSE ALARM PREVENTION GOLDEN RULE</h4>
        <p style="font-size:0.85rem; color:#374151;">${briefing.false_alarm_rule_of_thumb}</p>
      </div>
    `;
  } catch (err) {
    console.error("Error loading briefing:", err);
  }
}

// 7. Disaster Stress Testing
function runStressTest(scenarioType) {
  let preset = {};
  if (scenarioType === 'flash_flood') {
    preset = {
      water_level_m: 6.5,
      rainfall_mm_24h: 240,
      emergency_calls_6h: 25,
      road_closures: 4,
      power_outage_probability: 0.65,
      shelter_beds_available: 100,
      ambulances_available: 3
    };
  } else if (scenarioType === 'false_alarm') {
    preset = {
      water_level_m: 1.2,
      rainfall_mm_24h: 5,
      emergency_calls_6h: 40,
      road_closures: 0,
      power_outage_probability: 0.05,
      shelter_beds_available: 300,
      ambulances_available: 10
    };
  } else if (scenarioType === 'blackout') {
    preset = {
      water_level_m: 3.8,
      rainfall_mm_24h: 90,
      emergency_calls_6h: 18,
      road_closures: 6,
      power_outage_probability: 0.95,
      shelter_beds_available: 20,
      ambulances_available: 1
    };
  }

  // Set sliders
  for (const [key, val] of Object.entries(preset)) {
    const el = document.getElementById(key);
    if (el) {
      el.value = val;
      const valSpan = document.getElementById(key + '_val');
      if (valSpan) {
        valSpan.textContent = key === 'power_outage_probability' ? Math.round(val * 100) + '%' : val;
      }
    }
  }

  // Switch to simulator tab and trigger prediction
  document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector('[data-tab="panel-simulator"]').classList.add('active');
  document.getElementById('panel-simulator').classList.add('active');

  triggerPrediction();
}

function printBriefingSheet() {
  window.print();
}
