/** OptiRetain Dashboard — client-side rendering logic.
 *
 * Reads `customers.json` and renders:
 * 1. KPI summary cards (total, persuadable, recommended, avg uplift)
 * 2. Filterable/sortable customer table
 * 3. Per-customer detail panel with SHAP waterfall bars
 */

(function () {
    "use strict";

    // ── State ───────────────────────────────────────────────────────────────────────
    let allCustomers = [];
    let filteredRows = [];
    let currentSort = "expected_net_lift";

    // ── Init ──────────────────────────────────────────────────────────────────────────
    document.addEventListener("DOMContentLoaded", function () {
        loadDashboard();
        bindFilters();
    });

    async function loadDashboard() {
        try {
            const resp = await fetch("customers.json");
            const data = await resp.json();
            allCustomers = data.customers || [];
            renderKPIs(data.metadata);
            applyFilters();
        } catch (err) {
            document.getElementById("table-body").innerHTML =
                '<tr><td colspan="8" class="error">Failed to load dashboard data. Ensure customers.json is in the same directory.</td></tr>';
            console.error("Dashboard load error:", err);
        }
    }

    // ── KPI Cards ─────────────────────────────────────────────────────────────────────
    function renderKPIs(metadata) {
        const total = allCustomers.length;
        const persuadable = allCustomers.filter(c => c.segment === "Persuadable").length;
        const recommended = allCustomers.filter(c => c.recommended).length;

        const selected = allCustomers.filter(c => c.recommended);
        const avgUplift = selected.length > 0
            ? (selected.reduce((s, c) => s + c.uplift, 0) / selected.length)
            : 0;

        document.getElementById("kpi-total").textContent = total.toLocaleString();
        document.getElementById("kpi-persuadable").textContent = persuadable.toLocaleString();
        document.getElementById("kpi-recommended").textContent = recommended.toLocaleString();
        document.getElementById("kpi-avg-uplift").textContent = avgUplift.toFixed(4);
    }

    // ── Filters & Sort ────────────────────────────────────────────────────────────────
    function bindFilters() {
        document.getElementById("segment-filter").addEventListener("change", applyFilters);
        document.getElementById("search-input").addEventListener("input", applyFilters);
        document.getElementById("sort-select").addEventListener("change", function () {
            currentSort = this.value;
            applyFilters();
        });
    }

    function applyFilters() {
        const segmentVal = document.getElementById("segment-filter").value;
        const searchVal = document.getElementById("search-input").value.trim().toLowerCase();

        filteredRows = allCustomers.filter(c => {
            const matchSegment = segmentVal === "all" || c.segment === segmentVal;
            const matchSearch = !searchVal || c.customer_id.toLowerCase().includes(searchVal);
            return matchSegment && matchSearch;
        });

        // Sort descending by the selected metric.
        filteredRows.sort((a, b) => (b[currentSort] || 0) - (a[currentSort] || 0));

        renderTable();
    }

    // ── Table Rendering ───────────────────────────────────────────────────────────────
    function renderTable() {
        const tbody = document.getElementById("table-body");
        if (filteredRows.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="empty">No customers match the current filters.</td></tr>';
            return;
        }

        // Render up to 200 rows for performance.
        const displayRows = filteredRows.slice(0, 200);
        tbody.innerHTML = displayRows.map(c => {
            const pChurnColor = c.p_churn > 0.7 ? "#e74c3c" : c.p_churn > 0.5 ? "#f39c12" : "#2ecc71";
            const upliftDir = c.uplift >= 0 ? "+→" : "←-";
            const segColor = {
                "Persuadable": "#3498db",
                "Sure Thing": "#2ecc71",
                "Lost Cause": "#e67e22",
                "Sleeping Dog": "#e74c3c",
            }[c.segment] || "#95a5a6";

            return `<tr data-id="${escapeHtml(c.customer_id)}" onclick="showDetail('${escapeHtml(c.customer_id)}')">
                <td class="cust-id">${escapeHtml(c.customer_id)}</td>
                <td><span class="seg-badge seg-${c.segment.replace(/ /g, "_")}">${escapeHtml(c.segment)}</span></td>
                <td style="color:${pChurnColor};font-weight:600;">${(c.p_churn * 100).toFixed(1)}%</td>
                <td>${upliftDir} ${(Math.abs(c.uplift) * 100).toFixed(2)}%</td>
                <td>$${Number(c.clv).toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                <td>$${Number(c.cost).toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                <td style="font-weight:600;">$${Number(c.expected_net_lift).toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                <td>${c.recommended ? '<span class="rec-badge">Yes</span>' : '<span class="not-rec-badge">No</span>'}</td>
            </tr>`;
        }).join("");

        if (filteredRows.length > 200) {
            tbody.innerHTML += `<tr><td colspan="8" class="truncated">Showing 200 of ${filteredRows.length} rows — refine filters for full list.</td></tr>`;
        }
    }

    // ── Detail Panel (Waterfall) ──────────────────────────────────────────────────────
    window.showDetail = function (customerId) {
        const c = allCustomers.find(x => x.customer_id === customerId);
        if (!c) return;

        document.getElementById("detail-title").textContent = `Customer: ${c.customer_id}`;

        // Stats.
        document.getElementById("detail-stats").innerHTML = `
            <div class="stat-row"><span>Segment:</span><strong>${escapeHtml(c.segment)}</strong></div>
            <div class="stat-row"><span>P(Churn):</span><strong>${(c.p_churn * 100).toFixed(1)}%</strong></div>
            <div class="stat-row"><span>CATE:</span><strong>${(c.cate * 100).toFixed(2)}%</strong></div>
            <div class="stat-row"><span>Uplift:</span><strong style="color:${c.uplift > 0 ? "#27ae60" : "#e74c3c"}">${(c.uplift * 100).toFixed(2)}%</strong></div>
            <div class="stat-row"><span>CLV:</span><strong>$${Number(c.clv).toLocaleString()}</strong></div>
            <div class="stat-row"><span>Offer Cost:</span><strong>$${Number(c.cost).toLocaleString()}</strong></div>
            <div class="stat-row"><span>Net Lift:</span><strong style="color:${c.expected_net_lift > 0 ? "#27ae60" : "#e74c3c"}">$${Number(c.expected_net_lift).toLocaleString()}</strong></div>
        `;

        // SHAP Waterfall bars.
        const container = document.getElementById("waterfall-chart");
        const drivers = c.top_drivers || [];

        if (drivers.length === 0) {
            container.innerHTML = '<p class="no-drivers">No SHAP drivers available for this customer.</p>';
        } else {
            const maxAbs = Math.max(...drivers.map(d => Math.abs(d.shap)), 1e-6);
            container.innerHTML = drivers.map(d => {
                const widthPct = (Math.abs(d.shap) / maxAbs * 80).toFixed(1);
                const color = d.direction === "increases_risk" ? "#e74c3c" : "#2ecc71";
                const align = d.direction === "increases_risk" ? "right" : "left";
                return `<div class="bar-row ${d.direction}">
                    <span class="bar-label">${escapeHtml(d.feature)}: ${escapeHtml(String(d.value))}</span>
                    <div class="bar-track"><div class="bar-fill bar-${align}" style="width:${widthPct}%;background:${color}"></div></div>
                    <span class="bar-value">${d.shap > 0 ? "+" : ""}${d.shap.toFixed(4)}</span>
                </div>`;
            }).join("");
        }

        document.getElementById("detail-panel").classList.add("visible");
    };

    document.getElementById("detail-close").addEventListener("click", function () {
        document.getElementById("detail-panel").classList.remove("visible");
    });

    // ── Helpers ─────────────────────────────────────────────────────────────────────────
    function escapeHtml(str) {
        return String(str).replace(/[&<>"']/g, m => ({
            "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
        })[m]);
    }
})();
