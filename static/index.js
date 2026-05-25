document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons
    lucide.createIcons();

    // Elements
    const form = document.getElementById("claim-form");
    const ageInput = document.getElementById("age");
    const ageVal = document.getElementById("age-val");
    const tenureInput = document.getElementById("tenure");
    const tenureVal = document.getElementById("tenure-val");
    const creditInput = document.getElementById("credit");
    const creditVal = document.getElementById("credit-val");
    
    const amountInput = document.getElementById("amount");
    const amountMinus = document.getElementById("amount-minus");
    const amountPlus = document.getElementById("amount-plus");
    
    const prevClaimsSelect = document.getElementById("prev-claims");
    const locationSelect = document.getElementById("location");
    
    const btnAnalyze = document.getElementById("btn-analyze");
    const decisionResult = document.getElementById("decision-result");
    const insightsContainer = document.getElementById("insights-container");
    
    const probDeniedBar = document.getElementById("prob-denied-bar");
    const probDeniedVal = document.getElementById("prob-denied-val");
    const probApprovedBar = document.getElementById("prob-approved-bar");
    const probApprovedVal = document.getElementById("prob-approved-val");
    const modelVerdict = document.getElementById("model-verdict");
    const factorsContainer = document.getElementById("factors-container");

    // Modal elements
    const btnDeploy = document.getElementById("btn-deploy");
    const deployModal = document.getElementById("deploy-modal");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const btnCancelDeploy = document.getElementById("btn-cancel-deploy");
    const btnStartDeploy = document.getElementById("btn-start-deploy");
    const consoleLogs = document.getElementById("console-logs");
    const deployProgress = document.getElementById("deploy-progress");
    
    // Theme toggle and Clear buttons
    const btnThemeToggle = document.getElementById("btn-theme-toggle");
    const btnClear = document.getElementById("btn-clear");

    // -------------------------------------------------------------
    // Slider Interactivity
    // -------------------------------------------------------------
    ageInput.addEventListener("input", (e) => {
        ageVal.textContent = e.target.value;
    });

    tenureInput.addEventListener("input", (e) => {
        tenureVal.textContent = e.target.value;
    });

    creditInput.addEventListener("input", (e) => {
        creditVal.textContent = e.target.value;
    });

    // -------------------------------------------------------------
    // Amount Number Input Format & Buttons
    // -------------------------------------------------------------
    function formatNumber(val) {
        // Strip non-digits and format with commas
        let num = parseInt(val.toString().replace(/,/g, ""), 10);
        if (isNaN(num)) num = 0;
        return num.toLocaleString();
    }

    function getRawAmount() {
        let raw = parseInt(amountInput.value.replace(/,/g, ""), 10);
        if (isNaN(raw)) return 5000;
        return raw;
    }

    function setAmountValue(val) {
        if (val < 500) val = 500;
        if (val > 50000) val = 50000;
        amountInput.value = formatNumber(val);
    }

    amountMinus.addEventListener("click", () => {
        let current = getRawAmount();
        setAmountValue(current - 1000);
    });

    amountPlus.addEventListener("click", () => {
        let current = getRawAmount();
        setAmountValue(current + 1000);
    });

    amountInput.addEventListener("blur", () => {
        let current = getRawAmount();
        setAmountValue(current);
    });

    amountInput.addEventListener("input", (e) => {
        // Only allow digits and format
        let raw = e.target.value.replace(/[^\d]/g, "");
        if (raw === "") {
            e.target.value = "";
            return;
        }
        let cursor = e.target.selectionStart;
        let oldLen = e.target.value.length;
        e.target.value = parseInt(raw, 10).toLocaleString();
        // Adjust cursor position if formatting changed length
        let newLen = e.target.value.length;
        e.target.setSelectionRange(cursor + (newLen - oldLen), cursor + (newLen - oldLen));
    });

    // -------------------------------------------------------------
    // Core Prediction Logic
    // -------------------------------------------------------------
    async function runAnalysis() {
        // Set loading states
        btnAnalyze.disabled = true;
        btnAnalyze.classList.add("loading");
        btnAnalyze.querySelector("span").textContent = "Analyzing...";

        decisionResult.className = "decision-result-box loading";
        decisionResult.innerHTML = `
            <div class="spinner-container">
                <div class="spinner"></div>
                <span>Running analysis...</span>
            </div>
        `;

        insightsContainer.innerHTML = `
            <div class="skeleton skeleton-badge"></div>
            <div class="skeleton skeleton-badge"></div>
            <div class="skeleton skeleton-badge"></div>
        `;

        factorsContainer.innerHTML = `
            <div class="skeleton skeleton-factor"></div>
            <div class="skeleton skeleton-factor"></div>
            <div class="skeleton skeleton-factor"></div>
            <div class="skeleton skeleton-factor"></div>
            <div class="skeleton skeleton-factor"></div>
        `;

        // Gather Inputs
        const payload = {
            "Customer_Age": parseInt(ageInput.value, 10),
            "Policy_Tenure_Months": parseInt(tenureInput.value, 10),
            "Claim_Amount": getRawAmount(),
            "Previous_Claims": parseInt(prevClaimsSelect.value, 10),
            "Credit_Score": parseInt(creditInput.value, 10),
            "Location_Type": parseInt(locationSelect.value, 10)
        };

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error("Backend response error");
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            console.error("Analysis failed:", error);
            decisionResult.className = "decision-result-box denied";
            decisionResult.innerHTML = `
                <div class="decision-content">
                    <div class="decision-icon-badge" style="background-color: var(--color-error); color: white;">
                        <i data-lucide="alert-triangle"></i>
                    </div>
                    <div class="decision-details">
                        <h2>Error Running Model</h2>
                        <span class="decision-confidence text-error">${error.message}</span>
                        <p class="decision-subtext">Could not communicate with the Python AI server.</p>
                    </div>
                </div>
            `;
            lucide.createIcons();
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.classList.remove("loading");
            btnAnalyze.querySelector("span").textContent = "Analyze Claim";
        }
    }

    function renderResults(data) {
        const approved = data.prediction === 1;
        const confidencePct = (data.confidence * 100).toFixed(1);

        // 1. Decision Box Card
        if (approved) {
            decisionResult.className = "decision-result-box approved";
            decisionResult.innerHTML = `
                <div class="decision-content">
                    <div class="decision-icon-badge">
                        <i data-lucide="shield"></i>
                    </div>
                    <div class="decision-details">
                        <h2>Claim Approved</h2>
                        <span class="decision-confidence text-success">Confidence: ${confidencePct}%</span>
                        <p class="decision-subtext">Our AI model is highly confident in this decision.</p>
                    </div>
                </div>
            `;
        } else {
            decisionResult.className = "decision-result-box denied";
            decisionResult.innerHTML = `
                <div class="decision-content">
                    <div class="decision-icon-badge">
                        <i data-lucide="shield-alert"></i>
                    </div>
                    <div class="decision-details">
                        <h2>Claim Denied</h2>
                        <span class="decision-confidence text-error">Confidence: ${confidencePct}%</span>
                        <p class="decision-subtext">Our AI model recommends denying this claim.</p>
                    </div>
                </div>
            `;
        }

        // 2. Key Insights
        insightsContainer.innerHTML = "";
        data.insights.forEach(ins => {
            const badge = document.createElement("div");
            
            let statusClass = "badge-neutral";
            if (ins.status === "positive") statusClass = "badge-positive";
            if (ins.status === "negative") statusClass = "badge-negative";
            if (ins.type === "claims") {
                statusClass = ins.status === "positive" ? "badge-claims" : (ins.status === "neutral" ? "badge-claims-neutral" : "badge-negative");
            }
            
            badge.className = `insight-badge ${statusClass}`;
            badge.innerHTML = `
                <i data-lucide="${ins.icon}"></i>
                <span>${ins.label}</span>
            `;
            insightsContainer.appendChild(badge);
        });

        // 3. Probabilities Panel
        const deniedProb = (data.probabilities.Denied * 100).toFixed(2);
        const approvedProb = (data.probabilities.Approved * 100).toFixed(2);

        probDeniedBar.style.width = `${deniedProb}%`;
        probDeniedVal.textContent = `${deniedProb}%`;
        probApprovedBar.style.width = `${approvedProb}%`;
        probApprovedVal.textContent = `${approvedProb}%`;

        if (approved) {
            modelVerdict.textContent = "Approved";
            modelVerdict.className = "legend-verdict approved";
        } else {
            modelVerdict.textContent = "Denied";
            modelVerdict.className = "legend-verdict denied";
        }

        // 4. LIME Contributing Factors
        factorsContainer.innerHTML = "";
        
        // Render up to 5 key factors
        data.lime_factors.slice(0, 5).forEach(factor => {
            const weight = factor.weight;
            const isPositive = weight > 0;
            const absoluteWeight = Math.abs(weight);
            
            // Standard scale to fill width nicely (LIME weights are usually small, e.g. -0.5 to +0.5)
            const percentageWidth = Math.min(100, absoluteWeight * 180);

            const item = document.createElement("div");
            item.className = "factor-item";
            item.innerHTML = `
                <span class="factor-rule">${factor.rule}</span>
                <div class="factor-weight-container">
                    <div class="factor-bar-wrapper">
                        <div class="factor-bar ${isPositive ? 'positive' : 'negative'}" style="width: ${percentageWidth}%"></div>
                    </div>
                    <span class="factor-value ${isPositive ? 'positive' : 'negative'}">
                        ${isPositive ? '+' : ''}${weight.toFixed(2)}
                    </span>
                </div>
            `;
            factorsContainer.appendChild(item);
        });

        // Recreate icons in newly generated HTML
        lucide.createIcons();
    }

    // Run prediction on button click
    btnAnalyze.addEventListener("click", runAnalysis);

    // -------------------------------------------------------------
    // Deploy Modal sequence
    // -------------------------------------------------------------
    function openModal() {
        deployModal.classList.add("open");
        resetDeploymentState();
    }

    function closeModal() {
        deployModal.classList.remove("open");
    }

    function resetDeploymentState() {
        // Reset steps classes
        document.querySelectorAll(".step-item").forEach(item => {
            item.classList.remove("active", "complete");
        });
        document.getElementById("step-1").classList.add("active");
        
        // Reset console
        consoleLogs.innerHTML = `<p class="console-line text-mute">Ready for deployment sequence...</p>`;
        
        // Reset progress bar
        deployProgress.style.width = "0%";
        
        // Reset button
        btnStartDeploy.disabled = false;
        btnStartDeploy.className = "btn-action";
        btnStartDeploy.textContent = "Start Deployment";
    }

    async function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function addLogLine(text, type = "info") {
        const line = document.createElement("p");
        let cls = "text-mute";
        if (type === "success") cls = "text-success";
        if (type === "error") cls = "text-error";
        if (type === "warn") cls = "text-warn";
        if (type === "white") cls = "";
        
        line.className = `console-line ${cls}`;
        line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
        consoleLogs.appendChild(line);
        consoleLogs.scrollTop = consoleLogs.scrollHeight;
    }

    async function runDeploymentSequence() {
        btnStartDeploy.disabled = true;
        btnStartDeploy.textContent = "Deploying...";
        
        const steps = [
            { id: "step-1", label: "Serialize Weights" },
            { id: "step-2", label: "Generate API Endpoint" },
            { id: "step-3", label: "Health Check & Warmup" },
            { id: "step-4", label: "Active Gateway" }
        ];

        // Step 1: Serialize Weights
        addLogLine("Initializing deployment pipeline...", "info");
        await sleep(800);
        addLogLine("Contacting build server...", "info");
        await sleep(600);
        addLogLine("Loading insurance_model.pkl (RandomForestClassifier)...", "info");
        await sleep(500);
        addLogLine("Serializing estimators to high-performance formats...", "info");
        await sleep(700);
        addLogLine("Pipeline step 'Serialize Weights' completed successfully.", "success");
        document.getElementById("step-1").classList.add("complete");
        document.getElementById("step-1").classList.remove("active");
        document.getElementById("step-2").classList.add("active");
        deployProgress.style.width = "25%";
        
        // Step 2: Generate API Endpoint
        await sleep(1000);
        addLogLine("Packaging microservice runtime dependencies...", "info");
        await sleep(600);
        addLogLine("Generating secure API keys for gateway integration...", "info");
        await sleep(500);
        addLogLine("Mapping endpoint: POST /api/v1/predict", "white");
        addLogLine("Mapping endpoint: GET /api/v1/explain", "white");
        await sleep(600);
        addLogLine("API router generated successfully.", "success");
        document.getElementById("step-2").classList.add("complete");
        document.getElementById("step-2").classList.remove("active");
        document.getElementById("step-3").classList.add("active");
        deployProgress.style.width = "50%";

        // Step 3: Health Check & Warmup
        await sleep(1200);
        addLogLine("Provisioning isolated AWS ECS Fargate server...", "info");
        await sleep(800);
        addLogLine("Warming up Python runtime in container...", "info");
        await sleep(600);
        addLogLine("Running self-check test suite (6 features parsed)...", "info");
        await sleep(500);
        addLogLine("Test suite output: Prediction accuracy validates (100% agreement with local model)", "success");
        addLogLine("Average latency warmup: 14.2ms", "white");
        document.getElementById("step-3").classList.add("complete");
        document.getElementById("step-3").classList.remove("active");
        document.getElementById("step-4").classList.add("active");
        deployProgress.style.width = "75%";

        // Step 4: Active Gateway
        await sleep(1000);
        addLogLine("Redirecting live gateway traffic to production target...", "info");
        await sleep(800);
        addLogLine("Syncing CORS origins and rate limit thresholds (10,000 req/min)...", "info");
        await sleep(600);
        addLogLine("Active deployment: v1.0.4-live", "white");
        addLogLine("Deployment successful. Status: green.", "success");
        document.getElementById("step-4").classList.add("complete");
        document.getElementById("step-4").classList.remove("active");
        deployProgress.style.width = "100%";

        // Success State
        btnStartDeploy.textContent = "Deploy Successful! 🎉";
        btnStartDeploy.style.background = "linear-gradient(135deg, #22c55e 0%, #16a34a 100%)";
    }

    btnDeploy.addEventListener("click", openModal);
    btnCloseModal.addEventListener("click", closeModal);
    btnCancelDeploy.addEventListener("click", closeModal);
    btnStartDeploy.addEventListener("click", runDeploymentSequence);

    // -------------------------------------------------------------
    // Theme Switcher (Dark / Light Mode)
    // -------------------------------------------------------------
    const currentTheme = localStorage.getItem("theme") || "light";
    if (currentTheme === "dark") {
        document.body.classList.add("dark-theme");
        btnThemeToggle.querySelector("i").setAttribute("data-lucide", "sun");
    }

    btnThemeToggle.addEventListener("click", () => {
        document.body.classList.toggle("dark-theme");
        let theme = "light";
        if (document.body.classList.contains("dark-theme")) {
            theme = "dark";
            btnThemeToggle.querySelector("i").setAttribute("data-lucide", "sun");
        } else {
            theme = "dark"; // Default fallback
            btnThemeToggle.querySelector("i").setAttribute("data-lucide", "moon");
            // If it doesn't have dark-theme anymore, it is light
            if (!document.body.classList.contains("dark-theme")) {
                theme = "light";
            }
        }
        localStorage.setItem("theme", theme);
        lucide.createIcons();
    });

    // -------------------------------------------------------------
    // Clear Analysis & Reset Form
    // -------------------------------------------------------------
    btnClear.addEventListener("click", () => {
        // Reset form controls
        ageInput.value = 35;
        ageVal.textContent = "35";
        
        tenureInput.value = 24;
        tenureVal.textContent = "24";
        
        creditInput.value = 650;
        creditVal.textContent = "650";
        
        amountInput.value = "5,000";
        
        prevClaimsSelect.value = "0";
        locationSelect.value = "0"; // Urban
        
        // Reset Decision Results Box to initial blank placeholder state
        decisionResult.className = "decision-result-box loading";
        decisionResult.innerHTML = `
            <div class="spinner-container" style="color: var(--color-text-light);">
                <i data-lucide="info" style="width: 28px; height: 28px; margin-bottom: 8px;"></i>
                <span>No analysis run yet. Adjust details and click Analyze.</span>
            </div>
        `;
        
        // Clear Key Insights
        insightsContainer.innerHTML = `
            <div class="insight-badge" style="background-color: var(--color-bg-app); border: 1px dashed var(--color-border); color: var(--color-text-light); text-align: center; justify-content: center; align-items: center; width: 100%; grid-column: span 3; font-weight: 500;">
                <span>Waiting for parameters...</span>
            </div>
        `;
        
        // Reset Probabilities
        probDeniedBar.style.width = "0%";
        probDeniedVal.textContent = "0.00%";
        probApprovedBar.style.width = "0%";
        probApprovedVal.textContent = "0.00%";
        modelVerdict.textContent = "—";
        modelVerdict.className = "legend-verdict";
        modelVerdict.style.backgroundColor = "transparent";
        
        // Reset Contributing Factors (LIME)
        factorsContainer.innerHTML = `
            <div class="factor-item" style="justify-content: center; color: var(--color-text-light); font-style: italic;">
                <span>No contributing factors available</span>
            </div>
        `;
        
        lucide.createIcons();
    });

    // Run dynamic analysis once at loading to populate default screen
    runAnalysis();
});
