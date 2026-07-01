/**
 * SHL Assessment Intelligence — Frontend Application
 * Handles: Chat, Explorer, Evaluation, Developer Console
 */

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? `http://localhost:8000`
    : window.location.origin;

// ═══════════════════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════════════════

const state = {
    messages: [],
    turnCount: 0,
    catalog: [],
    filteredCatalog: [],
    isLoading: false,
    apiLogs: [],
    currentPage: 'chat',
};

// ═══════════════════════════════════════════════════════════════════════════
// DOM Elements
// ═══════════════════════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
    chatMessages: $('#chat-messages'),
    chatInput: $('#chat-input'),
    btnSend: $('#btn-send'),
    btnNewChat: $('#btn-new-chat'),
    turnCounter: $('#turn-counter'),
    contextBar: $('#context-bar'),
    contextPills: $('#context-pills'),
    reasoningContent: $('#reasoning-content'),
    statusDot: $('.status-dot'),
    statusText: $('.status-text'),
    explorerGrid: $('#explorer-grid'),
    explorerSearch: $('#explorer-search'),
    explorerStats: $('#explorer-stats'),
    filterType: $('#filter-type'),
    filterLevel: $('#filter-level'),
    apiLog: $('#api-log'),
    traceResults: $('#trace-results'),
    btnRunEval: $('#btn-run-eval'),
};

// ═══════════════════════════════════════════════════════════════════════════
// Navigation
// ═══════════════════════════════════════════════════════════════════════════

function initNavigation() {
    $$('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.dataset.page;
            switchPage(page);
        });
    });

    // Handle hash navigation
    const hash = window.location.hash.replace('#', '');
    if (hash) switchPage(hash);
}

function switchPage(page) {
    state.currentPage = page;

    // Update nav links
    $$('.nav-link').forEach(l => l.classList.remove('active'));
    const activeLink = $(`.nav-link[data-page="${page}"]`);
    if (activeLink) activeLink.classList.add('active');

    // Update pages
    $$('.page').forEach(p => p.classList.remove('page-active'));
    const activePage = $(`#page-${page}`);
    if (activePage) activePage.classList.add('page-active');

    // Load page data
    if (page === 'explorer' && state.catalog.length === 0) loadCatalog();

    window.location.hash = page;
}

// ═══════════════════════════════════════════════════════════════════════════
// Health Check
// ═══════════════════════════════════════════════════════════════════════════

async function checkHealth() {
    try {
        const resp = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
        const data = await resp.json();
        if (data.status === 'ok') {
            els.statusDot.className = 'status-dot connected';
            els.statusText.textContent = 'Connected';
            return true;
        }
    } catch (e) {
        els.statusDot.className = 'status-dot error';
        els.statusText.textContent = 'Offline';
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Chat
// ═══════════════════════════════════════════════════════════════════════════

function initChat() {
    // Input handling
    els.chatInput.addEventListener('input', () => {
        autoResizeInput();
        els.btnSend.disabled = !els.chatInput.value.trim();
        updateContextPills(els.chatInput.value);
    });

    els.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    els.btnSend.addEventListener('click', sendMessage);
    els.btnNewChat.addEventListener('click', resetChat);

    // Example chips
    $$('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            els.chatInput.value = chip.dataset.example;
            els.btnSend.disabled = false;
            autoResizeInput();
            sendMessage();
        });
    });
}

function autoResizeInput() {
    els.chatInput.style.height = 'auto';
    els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, 120) + 'px';
}

async function sendMessage() {
    const content = els.chatInput.value.trim();
    if (!content || state.isLoading) return;

    // Add user message
    state.messages.push({ role: 'user', content });
    state.turnCount++;
    renderMessage('user', content);

    // Clear input
    els.chatInput.value = '';
    els.btnSend.disabled = true;
    autoResizeInput();

    // Hide welcome
    const welcome = $('.chat-welcome');
    if (welcome) welcome.remove();

    // Show typing indicator
    showTypingIndicator();
    state.isLoading = true;

    // Call API
    const startTime = performance.now();
    try {
        const payload = { messages: state.messages };
        const resp = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(15000),
        });

        if (!resp.ok) {
            const errText = await resp.text().catch(() => 'Unknown error');
            throw new Error(`Server error ${resp.status}: ${errText}`);
        }

        const data = await resp.json();
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

        // Log to developer console
        logApiCall('POST /chat', payload, data, elapsed);

        // Add assistant message
        state.messages.push({ role: 'assistant', content: data.reply });
        state.turnCount++;

        // Remove typing indicator
        removeTypingIndicator();

        // Render assistant response
        renderAssistantMessage(data);

        // Update reasoning panel
        updateReasoningPanel(data, elapsed);

        // Update turn counter
        els.turnCounter.textContent = `Turn ${state.turnCount}/8`;

        // Update pipeline visualization
        animatePipeline();

    } catch (e) {
        removeTypingIndicator();
        let errorMsg = 'Sorry, I encountered an error. ';
        if (e.name === 'TimeoutError') {
            errorMsg += 'The request timed out. Please try again.';
        } else {
            errorMsg += e.message || 'Please check the server connection.';
        }
        renderMessage('assistant', errorMsg);
        logApiCall('POST /chat', { messages: state.messages }, { error: e.message }, '—');
    }

    state.isLoading = false;
}

function renderMessage(role, content) {
    const div = document.createElement('div');
    div.className = `message message-${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : 'AI';

    const bubble = document.createElement('div');
    bubble.className = 'message-content';
    bubble.textContent = content;

    div.appendChild(avatar);
    div.appendChild(bubble);
    els.chatMessages.appendChild(div);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function renderAssistantMessage(data) {
    const div = document.createElement('div');
    div.className = 'message message-assistant';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';

    const bubble = document.createElement('div');
    bubble.className = 'message-content';

    // Reply text
    const replyP = document.createElement('p');
    replyP.textContent = data.reply;
    bubble.appendChild(replyP);

    // Recommendations table
    if (data.recommendations && data.recommendations.length > 0) {
        const tableDiv = document.createElement('div');
        tableDiv.className = 'rec-table';

        let html = '<table><thead><tr><th>#</th><th>Assessment</th><th>Type</th></tr></thead><tbody>';
        data.recommendations.forEach((rec, i) => {
            const types = rec.test_type.split(',').map(t => t.trim());
            const typeBadges = types.map(t =>
                `<span class="rec-type rec-type-${t}">${t}</span>`
            ).join(' ');

            html += `<tr>
                <td>${i + 1}</td>
                <td><span class="rec-name"><a href="${rec.url}" target="_blank" rel="noopener">${rec.name}</a></span></td>
                <td>${typeBadges}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        tableDiv.innerHTML = html;
        bubble.appendChild(tableDiv);
    }

    // End of conversation badge
    if (data.end_of_conversation) {
        const badge = document.createElement('div');
        badge.style.cssText = 'margin-top:12px;font-size:12px;color:#22c55e;font-weight:600;';
        badge.textContent = '✓ Conversation complete';
        bubble.appendChild(badge);
    }

    div.appendChild(avatar);
    div.appendChild(bubble);
    els.chatMessages.appendChild(div);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'typing-indicator';
    div.id = 'typing-indicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.style.background = 'linear-gradient(135deg, #6366f1, #8b5cf6)';
    avatar.style.color = 'white';
    avatar.textContent = 'AI';

    const dots = document.createElement('div');
    dots.className = 'typing-dots';
    dots.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

    const stage = document.createElement('span');
    stage.className = 'typing-stage';
    stage.id = 'typing-stage';
    stage.textContent = 'Reconstructing context...';

    div.appendChild(avatar);
    div.appendChild(dots);
    div.appendChild(stage);
    els.chatMessages.appendChild(div);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;

    // Animate stages
    const stages = [
        'Reconstructing context...',
        'Analyzing intent...',
        'Searching catalog...',
        'Reranking results...',
        'Generating response...',
        'Verifying recommendations...',
    ];
    let i = 0;
    const interval = setInterval(() => {
        i++;
        if (i < stages.length) {
            const el = $('#typing-stage');
            if (el) el.textContent = stages[i];
        } else {
            clearInterval(interval);
        }
    }, 1500);

    // Store interval for cleanup
    div._interval = interval;
}

function removeTypingIndicator() {
    const indicator = $('#typing-indicator');
    if (indicator) {
        if (indicator._interval) clearInterval(indicator._interval);
        indicator.remove();
    }
}

function resetChat() {
    state.messages = [];
    state.turnCount = 0;
    els.chatMessages.innerHTML = `
        <div class="chat-welcome">
            <div class="welcome-icon">
                <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                    <circle cx="24" cy="24" r="23" stroke="url(#welcome-grad)" stroke-width="2" opacity="0.3"/>
                    <circle cx="24" cy="24" r="16" stroke="url(#welcome-grad)" stroke-width="2" opacity="0.5"/>
                    <circle cx="24" cy="24" r="8" fill="url(#welcome-grad)" opacity="0.8"/>
                    <defs>
                        <linearGradient id="welcome-grad" x1="0" y1="0" x2="48" y2="48">
                            <stop stop-color="#6366f1"/>
                            <stop offset="1" stop-color="#a78bfa"/>
                        </linearGradient>
                    </defs>
                </svg>
            </div>
            <h2>What role are you hiring for?</h2>
            <p>I'll help you find the right SHL assessments through conversation — clarifying requirements, comparing options, and building a grounded shortlist.</p>
            <div class="welcome-examples">
                <button class="example-chip" data-example="I'm hiring a senior Java developer who will work with Spring and AWS">Senior Java Developer</button>
                <button class="example-chip" data-example="We need to screen 500 entry-level contact centre agents for English US calls">Contact Centre Screening</button>
                <button class="example-chip" data-example="We need a solution for senior leadership — CXOs, directors with 15+ years experience">Executive Leadership</button>
                <button class="example-chip" data-example="Hiring graduate financial analysts — need numerical reasoning and finance knowledge tests">Graduate Financial Analysts</button>
            </div>
        </div>
    `;
    els.turnCounter.textContent = 'Turn 0/8';
    els.contextBar.style.display = 'none';
    els.contextPills.innerHTML = '';
    els.reasoningContent.innerHTML = '<div class="reasoning-empty"><p>Start a conversation to see the agent\'s reasoning process.</p></div>';

    // Re-attach example chip listeners
    $$('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            els.chatInput.value = chip.dataset.example;
            els.btnSend.disabled = false;
            autoResizeInput();
            sendMessage();
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// Context Pills (Live Intelligence)
// ═══════════════════════════════════════════════════════════════════════════

const SKILL_REGEX = /\b(java|python|javascript|typescript|angular|react|vue|node\.?js|c\+\+|c#|\.net|ruby|php|go|rust|swift|kotlin|scala|sql|mysql|postgres|mongodb|aws|azure|docker|kubernetes|spring|django|flask|linux|excel|word|networking|security|html|css|data science|machine learning|agile|scrum|ci\/cd|devops|git|sales|marketing|customer service|accounting|finance|medical|nursing|leadership|management)\b/gi;

const SENIORITY_REGEX = /\b(entry[- ]?level|junior|graduate|mid[- ]?level|senior|lead|manager|director|executive|cxo|vp|intern|trainee)\b/gi;

const ROLE_REGEX = /\b(developer|engineer|programmer|architect|analyst|designer|admin|assistant|operator|technician|nurse|doctor|agent|consultant|manager|leader|sales|recruiter)\b/gi;

function updateContextPills(text) {
    if (!text.trim()) {
        els.contextBar.style.display = 'none';
        return;
    }

    const pills = new Set();

    const skills = text.match(SKILL_REGEX);
    if (skills) skills.forEach(s => pills.add(`🔧 ${s}`));

    const seniority = text.match(SENIORITY_REGEX);
    if (seniority) seniority.forEach(s => pills.add(`📊 ${s}`));

    const roles = text.match(ROLE_REGEX);
    if (roles) roles.forEach(s => pills.add(`👤 ${s}`));

    if (pills.size > 0) {
        els.contextBar.style.display = 'block';
        els.contextPills.innerHTML = [...pills].map(p =>
            `<span class="context-pill">${p}</span>`
        ).join('');
    } else {
        els.contextBar.style.display = 'none';
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reasoning Panel
// ═══════════════════════════════════════════════════════════════════════════

function updateReasoningPanel(data, elapsed) {
    let html = '';

    // Intent
    html += `<div class="reasoning-section">
        <div class="reasoning-label">Response Time</div>
        <div class="reasoning-value">${elapsed}s</div>
    </div>`;

    // Recommendations count
    const recCount = data.recommendations ? data.recommendations.length : 0;
    html += `<div class="reasoning-section">
        <div class="reasoning-label">Recommendations</div>
        <div class="reasoning-value">${recCount} assessments</div>
    </div>`;

    // End of conversation
    html += `<div class="reasoning-section">
        <div class="reasoning-label">Conversation Status</div>
        <div class="reasoning-value">${data.end_of_conversation ? '✓ Complete' : '⟳ In progress'}</div>
    </div>`;

    // Turn info
    html += `<div class="reasoning-section">
        <div class="reasoning-label">Turn</div>
        <div class="reasoning-value">${state.turnCount} / 8</div>
        <div class="confidence-bar"><div class="confidence-fill" style="width:${(state.turnCount / 8) * 100}%"></div></div>
    </div>`;

    // Detected context from all messages
    const allUserText = state.messages.filter(m => m.role === 'user').map(m => m.content).join(' ');
    const detectedSkills = [...new Set((allUserText.match(SKILL_REGEX) || []).map(s => s.toLowerCase()))];
    const detectedSeniority = [...new Set((allUserText.match(SENIORITY_REGEX) || []).map(s => s.toLowerCase()))];

    if (detectedSkills.length > 0) {
        html += `<div class="reasoning-section">
            <div class="reasoning-label">Detected Skills</div>
            <div class="reasoning-value">${detectedSkills.map(s => `<span class="reasoning-badge badge-intent">${s}</span>`).join(' ')}</div>
        </div>`;
    }

    if (detectedSeniority.length > 0) {
        html += `<div class="reasoning-section">
            <div class="reasoning-label">Detected Seniority</div>
            <div class="reasoning-value">${detectedSeniority.map(s => `<span class="reasoning-badge badge-intent">${s}</span>`).join(' ')}</div>
        </div>`;
    }

    // Recommendation details
    if (data.recommendations && data.recommendations.length > 0) {
        html += `<div class="reasoning-section">
            <div class="reasoning-label">Assessment Types</div>
            <div class="reasoning-value">`;
        const types = {};
        data.recommendations.forEach(r => {
            r.test_type.split(',').forEach(t => {
                t = t.trim();
                types[t] = (types[t] || 0) + 1;
            });
        });
        const typeNames = { K: 'Knowledge', P: 'Personality', A: 'Aptitude', S: 'Simulation', B: 'SJT', C: 'Competency', D: 'Development' };
        html += Object.entries(types).map(([t, c]) =>
            `<span class="rec-type rec-type-${t}" style="margin-right:4px">${typeNames[t] || t}: ${c}</span>`
        ).join('');
        html += `</div></div>`;
    }

    els.reasoningContent.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════════════════
// Explorer
// ═══════════════════════════════════════════════════════════════════════════

async function loadCatalog() {
    try {
        const resp = await fetch(`${API_BASE}/api/catalog`);
        state.catalog = await resp.json();
        state.filteredCatalog = [...state.catalog];
        renderCatalog();
        els.explorerStats.textContent = `${state.catalog.length} assessments loaded`;
    } catch (e) {
        els.explorerStats.textContent = 'Failed to load catalog. Is the backend running?';
    }
}

function initExplorer() {
    els.explorerSearch.addEventListener('input', filterCatalog);
    els.filterType.addEventListener('change', filterCatalog);
    els.filterLevel.addEventListener('change', filterCatalog);
}

function filterCatalog() {
    const query = els.explorerSearch.value.toLowerCase();
    const type = els.filterType.value;
    const level = els.filterLevel.value;

    state.filteredCatalog = state.catalog.filter(p => {
        const matchesQuery = !query ||
            p.name.toLowerCase().includes(query) ||
            (p.description && p.description.toLowerCase().includes(query));
        const matchesType = !type || (p.keys && p.keys.includes(type));
        const matchesLevel = !level || (p.job_levels && p.job_levels.includes(level));
        return matchesQuery && matchesType && matchesLevel;
    });

    els.explorerStats.textContent = `${state.filteredCatalog.length} of ${state.catalog.length} assessments`;
    renderCatalog();
}

function renderCatalog() {
    const typeColors = {
        'Knowledge & Skills': 'rec-type-K',
        'Personality & Behavior': 'rec-type-P',
        'Ability & Aptitude': 'rec-type-A',
        'Simulations': 'rec-type-S',
        'Biodata & Situational Judgment': 'rec-type-B',
        'Competencies': 'rec-type-C',
        'Development & 360': 'rec-type-D',
    };

    const typeLabels = {
        'Knowledge & Skills': 'K',
        'Personality & Behavior': 'P',
        'Ability & Aptitude': 'A',
        'Simulations': 'S',
        'Biodata & Situational Judgment': 'B',
        'Competencies': 'C',
        'Development & 360': 'D',
    };

    els.explorerGrid.innerHTML = state.filteredCatalog.slice(0, 60).map(p => {
        const badges = (p.keys || []).map(k =>
            `<span class="card-badge ${typeColors[k] || ''}">${typeLabels[k] || k}</span>`
        ).join('');

        const desc = p.description ? p.description.substring(0, 150) + (p.description.length > 150 ? '...' : '') : '';

        return `<div class="assessment-card">
            <div class="card-header">
                <div class="card-name"><a href="${p.url}" target="_blank" rel="noopener">${p.name}</a></div>
            </div>
            <div class="card-badges">${badges}</div>
            <div class="card-description">${desc}</div>
            <div class="card-meta">
                ${p.duration ? `<span>⏱ ${p.duration}</span>` : ''}
                ${p.remote ? '<span>🌐 Remote</span>' : ''}
                ${p.adaptive ? '<span>⚡ Adaptive</span>' : ''}
            </div>
        </div>`;
    }).join('');

    if (state.filteredCatalog.length > 60) {
        els.explorerGrid.innerHTML += `<div class="assessment-card" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted)">
            + ${state.filteredCatalog.length - 60} more assessments. Use search to narrow results.
        </div>`;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluation
// ═══════════════════════════════════════════════════════════════════════════

const EVAL_TRACES = [
    { id: 'C1', name: 'Senior Leadership Selection', messages: [
        { role: 'user', content: 'We need a solution for senior leadership.' },
        { role: 'user', content: 'The pool consists of CXOs, director-level positions; people with more than 15 years of experience.' },
        { role: 'user', content: 'Selection — comparing candidates against a leadership benchmark.' },
    ], expected: ['Occupational Personality Questionnaire OPQ32r', 'OPQ Universal Competency Report 2.0', 'OPQ Leadership Report'] },
    { id: 'C2', name: 'Senior Rust Engineer', messages: [
        { role: 'user', content: "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?" },
        { role: 'user', content: 'Yes, go ahead. Should I also add a cognitive test for this level?' },
    ], expected: ['Smart Interview Live Coding', 'Linux Programming (General)', 'Networking and Implementation (New)', 'SHL Verify Interactive G+', 'Occupational Personality Questionnaire OPQ32r'] },
    { id: 'C3', name: 'Contact Centre Agents', messages: [
        { role: 'user', content: "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?" },
        { role: 'user', content: 'English.' },
        { role: 'user', content: 'US.' },
    ], expected: ['SVAR Spoken English (US) (New)', 'Contact Center Call Simulation (New)', 'Entry Level Customer Serv - Retail & Contact Center', 'Customer Service Phone Simulation'] },
    { id: 'C4', name: 'Graduate Financial Analysts', messages: [
        { role: 'user', content: 'Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test.' },
        { role: 'user', content: 'Good. Can you also add a situational judgement element — work-context decision making for graduates?' },
    ], expected: ['SHL Verify Interactive – Numerical Reasoning', 'Financial Accounting (New)', 'Basic Statistics (New)', 'Graduate Scenarios', 'Occupational Personality Questionnaire OPQ32r'] },
    { id: 'C5', name: 'Sales Re-skilling', messages: [
        { role: 'user', content: 'As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?' },
    ], expected: ['Global Skills Assessment', 'Global Skills Development Report', 'Occupational Personality Questionnaire OPQ32r', 'OPQ MQ Sales Report', 'Sales Transformation 2.0 - Individual Contributor'] },
    { id: 'C6', name: 'Plant Operators Safety', messages: [
        { role: 'user', content: "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?" },
    ], expected: ['Dependability and Safety Instrument (DSI)', 'Manufac. & Indust. - Safety & Dependability 8.0', 'Workplace Health and Safety (New)'] },
    { id: 'C7', name: 'Healthcare Admin Bilingual', messages: [
        { role: 'user', content: "We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical. What assessments work?" },
        { role: 'user', content: "They're functionally bilingual — English fluent for written work. Go with the hybrid." },
    ], expected: ['HIPAA (Security)', 'Medical Terminology (New)', 'Microsoft Word 365 - Essentials (New)', 'Dependability and Safety Instrument (DSI)', 'Occupational Personality Questionnaire OPQ32r'] },
    { id: 'C8', name: 'Admin Assistants Excel/Word', messages: [
        { role: 'user', content: 'I need to quickly screen admin assistants for Excel and Word daily.' },
        { role: 'user', content: 'In that case, I am OK with adding a simulation - we want to capture the capabilities.' },
    ], expected: ['Microsoft Excel 365 (New)', 'Microsoft Word 365 (New)', 'MS Excel (New)', 'MS Word (New)', 'Occupational Personality Questionnaire OPQ32r'] },
    { id: 'C9', name: 'Senior Full-Stack Engineer', messages: [
        { role: 'user', content: 'Here\'s the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker. Will own end-to-end microservice delivery, contribute to architectural decisions, and mentor mid-level engineers. Strong CI/CD and cloud-native experience required."' },
        { role: 'user', content: 'Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant. Angular is occasional.' },
        { role: 'user', content: 'Senior IC. They lead design on their own services but don\'t manage other engineers directly.' },
    ], expected: ['Core Java (Advanced Level) (New)', 'Spring (New)', 'SQL (New)', 'Amazon Web Services (AWS) Development (New)', 'Docker (New)', 'SHL Verify Interactive G+', 'Occupational Personality Questionnaire OPQ32r'] },
    { id: 'C10', name: 'Graduate Management Trainees', messages: [
        { role: 'user', content: 'We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates.' },
    ], expected: ['SHL Verify Interactive G+', 'Occupational Personality Questionnaire OPQ32r', 'Graduate Scenarios'] },
];

function initEval() {
    els.btnRunEval.addEventListener('click', runAllTraces);
}

async function runAllTraces() {
    els.btnRunEval.disabled = true;
    els.btnRunEval.textContent = 'Running...';
    els.traceResults.innerHTML = '';

    const results = [];
    for (const trace of EVAL_TRACES) {
        const result = await runTrace(trace);
        results.push(result);
        renderTraceResult(result);
    }

    // Update metrics
    const avgRecall = results.reduce((s, r) => s + r.recall, 0) / results.length;
    const schemaPass = results.filter(r => r.schemaOk).length / results.length;
    const avgLatency = results.reduce((s, r) => s + r.latency, 0) / results.length;

    $('#metric-recall').textContent = (avgRecall * 100).toFixed(1) + '%';
    $('#metric-schema').textContent = (schemaPass * 100).toFixed(0) + '%';
    $('#metric-probes').textContent = '—';
    $('#metric-latency').textContent = avgLatency.toFixed(1) + 's';

    $('#recall-fill').style.width = (avgRecall * 100) + '%';
    $('#schema-fill').style.width = (schemaPass * 100) + '%';
    $('#latency-fill').style.width = Math.max(0, 100 - avgLatency * 10) + '%';

    els.btnRunEval.disabled = false;
    els.btnRunEval.textContent = 'Run All Traces';
}

async function runTrace(trace) {
    const messages = [];
    let lastData = null;
    const startTime = performance.now();

    for (const msg of trace.messages) {
        messages.push(msg);
        if (lastData && lastData.reply) {
            messages.push({ role: 'assistant', content: lastData.reply });
        }
        messages.push(msg);
        // Rebuild properly - we need to interleave
        break; // Simplified: send first message and see result
    }

    // Actually, let's do a proper multi-turn replay
    const convMessages = [];
    let result = null;

    for (const msg of trace.messages) {
        convMessages.push(msg);
        try {
            const resp = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages: convMessages }),
                signal: AbortSignal.timeout(35000),
            });
            result = await resp.json();
            convMessages.push({ role: 'assistant', content: result.reply });
        } catch (e) {
            result = { reply: 'Error', recommendations: null, end_of_conversation: false };
            break;
        }
    }

    const elapsed = (performance.now() - startTime) / 1000;

    // Calculate recall
    const recommended = (result.recommendations || []).map(r => r.name);
    const recall = calculateRecall(recommended, trace.expected);

    // Schema check
    const schemaOk = result && typeof result.reply === 'string' &&
        (result.recommendations === null || Array.isArray(result.recommendations)) &&
        typeof result.end_of_conversation === 'boolean';

    return {
        id: trace.id,
        name: trace.name,
        recall,
        schemaOk,
        latency: elapsed,
        recommended,
        expected: trace.expected,
    };
}

function calculateRecall(recommended, relevant) {
    if (!relevant || relevant.length === 0) return 0;
    const recSet = new Set(recommended.map(r => r.toLowerCase().trim()));
    const hits = relevant.filter(r => recSet.has(r.toLowerCase().trim())).length;
    return hits / relevant.length;
}

function renderTraceResult(result) {
    const div = document.createElement('div');
    div.className = 'trace-card';

    const recallPct = (result.recall * 100).toFixed(0);
    const color = result.recall >= 0.7 ? 'var(--success)' : result.recall >= 0.4 ? 'var(--warning)' : 'var(--error)';

    div.innerHTML = `
        <div>
            <div class="trace-name">${result.id}: ${result.name}</div>
            <div style="font-size:12px;color:var(--text-muted);margin-top:4px">${result.recommended.length} recs, ${result.latency.toFixed(1)}s</div>
        </div>
        <div class="trace-score" style="color:${color}">
            ${result.schemaOk ? '✓' : '✗'} Recall: ${recallPct}%
        </div>
    `;
    els.traceResults.appendChild(div);
}

// ═══════════════════════════════════════════════════════════════════════════
// Developer Console
// ═══════════════════════════════════════════════════════════════════════════

function logApiCall(endpoint, request, response, elapsed) {
    const entry = { endpoint, request, response, elapsed, timestamp: new Date().toISOString() };
    state.apiLogs.unshift(entry);
    if (state.apiLogs.length > 20) state.apiLogs.pop();

    const logHtml = state.apiLogs.map(log => {
        const req = JSON.stringify(log.request, null, 2).substring(0, 500);
        const res = JSON.stringify(log.response, null, 2).substring(0, 500);
        return `// ${log.timestamp} — ${log.endpoint} (${log.elapsed}s)\n// Request:\n${req}\n// Response:\n${res}\n${'─'.repeat(60)}`;
    }).join('\n\n');

    els.apiLog.innerHTML = `<pre><code>${logHtml}</code></pre>`;
}

function animatePipeline() {
    const stages = $$('.stage-node');
    stages.forEach(s => s.classList.remove('active'));

    const stageNames = ['Reconstruct', 'Route Intent', 'Retrieve', 'Recommend', 'Verify', 'Output'];
    let i = 0;
    const interval = setInterval(() => {
        if (i > 0) stages[i - 1].classList.remove('active');
        if (i < stages.length) {
            stages[i].classList.add('active');
            i++;
        } else {
            clearInterval(interval);
            setTimeout(() => stages.forEach(s => s.classList.remove('active')), 1000);
        }
    }, 300);
}

// ═══════════════════════════════════════════════════════════════════════════
// Initialize
// ═══════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initChat();
    initExplorer();
    initEval();
    checkHealth();

    // Periodic health check
    setInterval(checkHealth, 30000);
});
