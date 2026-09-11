document.addEventListener('DOMContentLoaded', () => {
  runMLAnalysis();
  initDragAndDrop();
  runNLPAnalysis();
  initVoiceRecognition();
  runSLMSummarization();
});

// Section Tab Switcher
function switchTab(tabId) {
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-target') === tabId);
  });
  
  document.querySelectorAll('.section-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === tabId);
  });
}

// -----------------------------------------------------------------------------
// STAGE 01: MACHINE LEARNING (ML) ANALYSIS
// -----------------------------------------------------------------------------
function runMLAnalysis() {
  const water = parseFloat(document.getElementById('ml_water_level').value);
  const rain = parseFloat(document.getElementById('ml_rainfall').value);
  const calls = parseFloat(document.getElementById('ml_calls').value);
  const roads = parseFloat(document.getElementById('ml_roads').value);

  // Update label displays
  document.getElementById('val_water_level').innerText = `${water.toFixed(1)} m`;
  document.getElementById('val_rainfall').innerText = `${rain.toFixed(0)} mm`;
  document.getElementById('val_calls').innerText = `${calls.toFixed(0)} calls`;
  document.getElementById('val_roads').innerText = `${roads.toFixed(0)} ${roads === 1 ? 'road' : 'roads'}`;

  // Call ML Prediction API
  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      water_level_m: water,
      rainfall_mm_24h: rain,
      emergency_calls_6h: calls,
      road_closures: roads
    })
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === 'success') {
      updateMLUI(data);
    }
  })
  .catch(err => console.error("ML Prediction Error:", err));
}

function updateMLUI(data) {
  // Score
  document.getElementById('ml_score_val').innerHTML = `${data.risk_score} <span style="font-size: 1rem; color: var(--text-muted);">/ 100</span>`;

  // Risk Badge
  const badge = document.getElementById('ml_risk_badge');
  const level = (data.risk_level || 'Low').toUpperCase();
  badge.innerText = `${level} RISK`;

  if (level === 'SEVERE') {
    badge.className = 'risk-badge badge-severe';
  } else if (level === 'MODERATE') {
    badge.className = 'risk-badge badge-mod';
  } else {
    badge.className = 'risk-badge badge-low';
  }

  // Probabilities
  const probLow = Math.round((data.probabilities.Low || 0) * 100);
  const probMod = Math.round((data.probabilities.Moderate || 0) * 100);
  const probSev = Math.round((data.probabilities.Severe || 0) * 100);

  document.getElementById('prob_low_text').innerText = `${probLow}%`;
  document.getElementById('prob_low_bar').style.width = `${probLow}%`;

  document.getElementById('prob_mod_text').innerText = `${probMod}%`;
  document.getElementById('prob_mod_bar').style.width = `${probMod}%`;

  document.getElementById('prob_sev_text').innerText = `${probSev}%`;
  document.getElementById('prob_sev_bar').style.width = `${probSev}%`;

  // Action Text
  document.getElementById('ml_action_text').innerText = data.tactical_action || "GREEN CLEAR: Routine monitoring.";
}


// -----------------------------------------------------------------------------
// STAGE 02: DEEP LEARNING (DL) IMAGE CLASSIFIER
// -----------------------------------------------------------------------------
function initDragAndDrop() {
  const dropzone = document.querySelector('.dropzone');
  if (!dropzone) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-over');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-over');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      processImageFile(files[0]);
    }
  });
}

function handleDLUpload(event) {
  const file = event.target.files[0];
  if (file) {
    processImageFile(file);
  }
}

function processImageFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const previewWrapper = document.getElementById('preview_wrapper');
    const previewImg = document.getElementById('dl_img_preview');
    if (previewWrapper) previewWrapper.style.display = 'block';
    if (previewImg) previewImg.src = e.target.result;
  };
  reader.readAsDataURL(file);

  const formData = new FormData();
  formData.append('file', file);

  fetch('/api/dl/predict_image', {
    method: 'POST',
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === 'success') {
      updateDLUI(data);
    }
  })
  .catch(err => console.error("DL Image Prediction Error:", err));
}

function updateDLUI(data) {
  const label = data.predicted_label || "NOT FLOODED";
  const confidence = data.confidence_percentage || 95.0;
  const isFlooded = data.is_flooded || label === 'FLOODED';

  const statusBadge = document.getElementById('dl_status_badge');
  statusBadge.innerText = label;
  statusBadge.className = isFlooded ? 'risk-badge badge-severe' : 'risk-badge badge-low';

  document.getElementById('dl_confidence_display').innerText = `${confidence}%`;
  const confBar = document.getElementById('dl_confidence_bar');
  confBar.style.width = `${confidence}%`;
  confBar.className = isFlooded ? 'confidence-fill fill-sev' : 'confidence-fill fill-low';

  const probFlooded = data.prob_flooded || (isFlooded ? confidence : (100 - confidence));
  const probClear = data.prob_clear || (isFlooded ? (100 - confidence) : confidence);

  document.getElementById('dl_prob_flooded_text').innerText = `${probFlooded}%`;
  document.getElementById('dl_prob_flooded_bar').style.width = `${probFlooded}%`;

  document.getElementById('dl_prob_clear_text').innerText = `${probClear}%`;
  document.getElementById('dl_prob_clear_bar').style.width = `${probClear}%`;

  const summaryElem = document.getElementById('dl_summary_text');
  summaryElem.innerHTML = `The PyTorch FloodResNet Neural Network evaluated the uploaded image and predicted <strong>${label}</strong> with <strong>${confidence}% confidence</strong>.`;
}


// -----------------------------------------------------------------------------
// STAGE 03: DAY 3 NATURAL LANGUAGE PROCESSING (NLP)
// -----------------------------------------------------------------------------
function loadNLPPreset(msgText, btnElement) {
  const inputElem = document.getElementById('nlp_message_input');
  if (inputElem) {
    inputElem.value = msgText;
  }

  if (btnElement) {
    document.querySelectorAll('#nlp_preset_group .btn-sample').forEach(b => b.classList.remove('active'));
    btnElement.classList.add('active');
  }

  runNLPAnalysis();
}

function runNLPAnalysis() {
  const inputElem = document.getElementById('nlp_message_input');
  let msgText = inputElem ? inputElem.value.trim() : '';

  if (!msgText) {
    return; // Don't run analysis on empty text, wait for user input/speech!
  }

  fetch('/api/nlp/predict_message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: msgText })
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === 'success') {
      updateNLPUI(data);
    }
  })
  .catch(err => console.error("NLP Prediction Error:", err));
}

function updateNLPUI(data) {
  const urgency = (data.predicted_urgency || 'LOW').toUpperCase();
  const urgencyConf = data.urgency_confidence || 95.0;
  const disasterType = data.predicted_disaster_type || 'Flood';
  const disasterConf = data.disaster_confidence || 90.0;

  // 1. Urgency Badge
  const badge = document.getElementById('nlp_urgency_badge');
  badge.innerText = urgency;
  if (urgency === 'HIGH' || urgency === 'SEVERE') {
    badge.className = 'risk-badge badge-severe';
  } else if (urgency === 'MODERATE') {
    badge.className = 'risk-badge badge-mod';
  } else {
    badge.className = 'risk-badge badge-low';
  }

  document.getElementById('nlp_urgency_conf').innerText = `${urgencyConf}%`;

  // 2. Disaster Type Badge
  document.getElementById('nlp_disaster_type').innerText = disasterType;
  document.getElementById('nlp_disaster_conf').innerText = `${disasterConf}%`;

  // 3. Extracted Entities
  document.getElementById('nlp_location_val').innerText = data.location || 'Unknown Location';
  document.getElementById('nlp_people_val').innerText = data.people_affected || '0';
  document.getElementById('nlp_resource_val').innerText = data.resource_required || 'General Relief Assistance';

  // 4. Similarity Match
  const simMatch = data.historical_similarity_percentage || 84.5;
  document.getElementById('nlp_similarity_val').innerText = `${simMatch}% Match`;
  const simBar = document.getElementById('nlp_similarity_bar');
  simBar.style.width = `${simMatch}%`;

  const simMsg = data.most_similar_historical_message || 'No matching log';
  document.getElementById('nlp_similar_msg').innerText = `"${simMsg.substring(0, 70)}${simMsg.length > 70 ? '...' : ''}"`;

  // 5. Recommended Action
  document.getElementById('nlp_action_val').innerText = data.recommended_action || "🟢 GREEN CLEAR: Maintain standard standby.";
}

// -----------------------------------------------------------------------------
// VOICE-TO-TEXT & AUDIO FILE TRANSCRIPTION ENGINE
// -----------------------------------------------------------------------------
let recognition = null;
let isRecording = false;

function initVoiceRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn("Web Speech API not supported in this browser.");
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    isRecording = true;
    updateVoiceUI(true, "Listening to live microphone... Speak your emergency message now!");
  };

  recognition.onresult = (event) => {
    let interimTranscript = '';
    let finalTranscript = '';

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript;
      } else {
        interimTranscript += event.results[i][0].transcript;
      }
    }

    const currentText = finalTranscript || interimTranscript;
    if (currentText) {
      document.getElementById('nlp_message_input').value = currentText;
      
      const previewBox = document.getElementById('transcript_preview_box');
      const previewText = document.getElementById('transcript_preview_text');
      if (previewBox && previewText) {
        previewBox.style.display = 'flex';
        previewText.innerText = `Recognized Live Voice: "${currentText}"`;
      }
      
      runNLPAnalysis();
    }
  };

  recognition.onerror = (event) => {
    console.error("Speech Recognition Error:", event.error);
    updateVoiceUI(false);
    const previewText = document.getElementById('transcript_preview_text');
    if (previewText) {
      previewText.innerText = `⚠️ Voice Recognition Note: ${event.error}. You can also click sample voice clips below or type your message.`;
    }
  };

  recognition.onend = () => {
    isRecording = false;
    updateVoiceUI(false);
  };
}

function toggleVoiceRecording() {
  if (!recognition) {
    initVoiceRecognition();
  }

  if (!recognition) {
    alert("Speech Recognition is not supported by your browser. Please use Chrome, Edge, or Safari, or click one of the sample voice clips.");
    return;
  }

  if (isRecording) {
    recognition.stop();
    isRecording = false;
    updateVoiceUI(false);
  } else {
    try {
      // Clear input so previous text is cleared!
      const inputElem = document.getElementById('nlp_message_input');
      if (inputElem) inputElem.value = '';
      
      const previewBox = document.getElementById('transcript_preview_box');
      const previewText = document.getElementById('transcript_preview_text');
      if (previewBox && previewText) {
        previewBox.style.display = 'flex';
        previewText.innerText = 'Listening to microphone... Speak clearly into your mic...';
      }

      recognition.start();
    } catch (err) {
      console.error(err);
      updateVoiceUI(false);
    }
  }
}

function updateVoiceUI(active, statusMessage) {
  const btn = document.getElementById('btn_start_voice');
  const btnText = document.getElementById('mic_btn_text');
  const micIcon = document.getElementById('mic_icon');
  const banner = document.getElementById('voice_status_banner');
  const statusText = document.getElementById('voice_status_text');

  if (active) {
    if (btn) btn.classList.add('recording');
    if (btnText) btnText.innerText = "Stop Recording";
    if (micIcon) micIcon.className = "fa-solid fa-microphone-slash";
    if (banner) banner.style.display = 'flex';
    if (statusText && statusMessage) statusText.innerText = statusMessage;
  } else {
    if (btn) btn.classList.remove('recording');
    if (btnText) btnText.innerText = "Start Voice Recording";
    if (micIcon) micIcon.className = "fa-solid fa-microphone";
    if (banner) banner.style.display = 'none';
  }
}

function handleAudioUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  const banner = document.getElementById('voice_status_banner');
  const statusText = document.getElementById('voice_status_text');
  if (banner) banner.style.display = 'flex';
  if (statusText) statusText.innerText = `Processing audio file "${file.name}"...`;

  fetch('/api/nlp/voice_to_text', {
    method: 'POST',
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    if (banner) banner.style.display = 'none';
    if (data.status === 'success' || data.transcribed_text) {
      const transcribed = data.transcribed_text || "Emergency assistance requested";
      document.getElementById('nlp_message_input').value = transcribed;
      
      const previewBox = document.getElementById('transcript_preview_box');
      const previewText = document.getElementById('transcript_preview_text');
      if (previewBox && previewText) {
        previewBox.style.display = 'flex';
        previewText.innerText = `Transcribed from ${file.name}: "${transcribed}"`;
      }
      
      updateNLPUI(data);
    }
  })
  .catch(err => {
    if (banner) banner.style.display = 'none';
    console.error("Audio Upload Speech Error:", err);
  });
}

function transcribeSampleAudio(filename, text) {
  const inputElem = document.getElementById('nlp_message_input');
  if (inputElem) {
    inputElem.value = text;
  }
  const previewBox = document.getElementById('transcript_preview_box');
  const previewText = document.getElementById('transcript_preview_text');
  if (previewBox && previewText) {
    previewBox.style.display = 'flex';
    previewText.innerText = `Transcribed audio file [${filename}]: "${text}"`;
  }
  runNLPAnalysis();
}

// -----------------------------------------------------------------------------
// STAGE 04: SMALL LANGUAGE MODEL (SLM) REPORT SUMMARIZATION ENGINE
// -----------------------------------------------------------------------------
function runSLMSummarization() {
  const reportInput = document.getElementById('slm_report_input');
  let reportText = reportInput ? reportInput.value.trim() : '';

  const badge = document.getElementById('slm_word_count_badge');
  if (!reportText) {
    if (badge) badge.innerText = "0 words";
    resetSLMUI();
    return;
  }

  // Update word count badge
  const words = reportText.split(/\s+/).filter(w => w.length > 0).length;
  if (badge) badge.innerText = `${words} words`;

  fetch('/api/slm/summarize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ report: reportText })
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === 'success') {
      updateSLMUI(data);
    }
  })
  .catch(err => console.error("SLM Summarization Error:", err));
}

function resetSLMUI() {
  document.getElementById('slm_tier_label').innerText = '📋 AI ADAPTIVE BRIEFING TIER';
  const badge = document.getElementById('slm_urgency_badge');
  if (badge) {
    badge.innerText = 'PENDING';
    badge.className = 'risk-badge badge-low';
  }
  document.getElementById('slm_briefing_text').innerText = 'Paste a disaster report to generate a severity-adaptive summary (High: 2-Sentence Executive Brief | Moderate: 1-Sentence Field Brief | Low: 1-Line Compact Brief).';
  document.getElementById('slm_fact_location').innerText = 'Pending Report Input';
  document.getElementById('slm_fact_headcount').innerText = '0';
  document.getElementById('slm_compression_val').innerText = '0%';
  document.getElementById('slm_latency_val').innerText = '0.00 ms';
}

function updateSLMUI(data) {
  const adaptive = data.adaptive_briefing || {};
  const facts = data.extracted_facts || {};
  const metrics = data.metrics || {};

  // Tier Label & Briefing Text
  document.getElementById('slm_tier_label').innerText = adaptive.tier_label || '📋 AI ADAPTIVE BRIEFING';
  document.getElementById('slm_briefing_text').innerText = adaptive.briefing_text || 'Report summary generated.';

  // Urgency Badge
  const urgency = (facts.urgency || 'MODERATE').toUpperCase();
  const urgencyBadge = document.getElementById('slm_urgency_badge');
  if (urgencyBadge) {
    urgencyBadge.innerText = urgency;
    urgencyBadge.className = adaptive.badge_class || (urgency === 'SEVERE' || urgency === 'HIGH' ? 'risk-badge badge-severe' : (urgency === 'MODERATE' ? 'risk-badge badge-mod' : 'risk-badge badge-low'));
  }

  // Fact Retention Matrix
  document.getElementById('slm_fact_location').innerText = facts.location || 'Unknown Area';
  document.getElementById('slm_fact_headcount').innerText = facts.headcount || '0';

  // Metrics
  document.getElementById('slm_compression_val').innerText = `${metrics.compression_ratio_percentage || 0}%`;
  document.getElementById('slm_latency_val').innerText = `${metrics.inference_latency_ms || 0.13} ms`;
}



