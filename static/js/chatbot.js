/* =============================================
   Bangkok Safety Assistant — Chat Logic
   ============================================= */

(function () {
    "use strict";

    // DOM refs
    const pageEl = document.getElementById("assistantPage");
    const shellEl = document.getElementById("chatShell");
    const messagesEl = document.getElementById("chatMessages");
    const inputEl = document.getElementById("chatInput");
    const sendBtn = document.getElementById("chatSend");
    const suggestionsEl = document.getElementById("chatSuggestions");
    const sugButtonsEl = document.getElementById("suggestionsButtons");
    const typingEl = document.getElementById("typingIndicator");
    const composerEl = document.getElementById("assistantComposer");
    const fileButtonEl = document.getElementById("chatFileButton");
    const fileInputEl = document.getElementById("chatFileInput");
    const accordionBtn = document.getElementById("howItWorksToggle");
    const accordionBody = document.getElementById("howItWorksContent");

    let isSending = false;
    let history = [];

    // ------------------------------------------------------------------
    // Suggestions
    // ------------------------------------------------------------------
    async function loadSuggestions() {
        try {
            const res = await fetch("/api/chatbot/suggestions");
            const data = await res.json();
            if (data.success && data.suggestions) {
                renderSuggestions(data.suggestions);
            }
        } catch (e) {
            // Hardcoded fallback
            renderSuggestions([
                "What causes most accidents?",
                "Which areas are most dangerous?",
                "What time are accidents most common?",
                "Is this route risky tonight?",
            ]);
        }
    }

    function renderSuggestions(items) {
        sugButtonsEl.innerHTML = "";
        items.slice(0, 4).forEach(function (text) {
            var btn = document.createElement("button");
            btn.className = "suggestion-btn";
            btn.textContent = text;
            btn.addEventListener("click", function () {
                sendMessage(text);
            });
            sugButtonsEl.appendChild(btn);
        });
    }

    function prefersReducedMotion() {
        return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    }

    function hideSuggestions() {
        if (!suggestionsEl) return;
        suggestionsEl.classList.add("is-hidden");
    }

    function setConversationStarted() {
        if (shellEl) shellEl.classList.add("has-messages");
        if (pageEl) pageEl.classList.add("has-messages");
    }

    function updateComposerOffset() {
        if (!composerEl || !pageEl) return;
        pageEl.style.setProperty("--assistant-composer-offset", composerEl.offsetHeight + "px");
    }

    // ------------------------------------------------------------------
    // Messages
    // ------------------------------------------------------------------
    function addMessage(role, content, meta) {
        var wasCollapsed = shellEl && !shellEl.classList.contains("has-messages");
        var wrapper = document.createElement("div");
        wrapper.className = "message " + role;

        var avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.textContent = role === "bot" ? "SD" : "ME";
        avatar.setAttribute("aria-hidden", "true");

        var contentDiv = document.createElement("div");
        contentDiv.className = "message-content";

        var textDiv = document.createElement("div");
        textDiv.className = "message-text";
        textDiv.innerHTML = formatContent(content, meta);
        contentDiv.appendChild(textDiv);

        // Source badge for bot messages
        if (role === "bot" && meta && meta.source) {
            var badge = document.createElement("span");
            badge.className = "message-source " + meta.source;
            var labels = {
                scripted: "Instant answer",
                groq: "AI-powered",
                error: "Fallback",
            };
            badge.textContent = labels[meta.source] || meta.source;
            contentDiv.appendChild(badge);
        }

        wrapper.appendChild(avatar);
        wrapper.appendChild(contentDiv);
        messagesEl.appendChild(wrapper);
        setConversationStarted();
        scrollToBottom(true, wrapper);
        if (wasCollapsed && !prefersReducedMotion()) {
            window.setTimeout(function () {
                scrollToBottom(true, wrapper);
            }, 220);
        }
    }

    function formatContent(content, meta) {
        if (!meta) return "<p>" + escapeHtml(content) + "</p>";

        // Error type
        if (meta.type === "error") {
            return "<p>" + escapeHtml(meta.message || content) + "</p>";
        }

        var out = formatMarkdown(meta.message || content || "");
        var followup = meta.followup || meta.follow_up;
        if (followup) {
            out += '<div class="message-followup">' + formatMarkdown(followup) + "</div>";
        }
        if (Array.isArray(meta.actions) && meta.actions.length) {
            out += renderActionButtons(meta.actions);
        }
        return out;
    }

    // Minimal markdown-like formatting for AI responses
    function formatMarkdown(text) {
        var html = escapeHtml(text);
        // Bold
        html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        // Bullet lists (lines starting with - or *)
        html = html.replace(/^[\-\*]\s+(.+)$/gm, "<li>$1</li>");
        // Wrap consecutive <li> in <ul>
        html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");
        // Numbered lists
        html = html.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");
        html = html.replace(
            /(<li>(?:(?!<\/?ul>).)*<\/li>\n?)+/g,
            function (match) {
                if (match.indexOf("<ul>") === -1 && /<li>/.test(match)) {
                    return "<ol>" + match + "</ol>";
                }
                return match;
            }
        );
        // Paragraphs (double newline)
        html = html.replace(/\n{2,}/g, "</p><p>");
        // Single newlines outside lists
        html = html.replace(/\n/g, "<br>");
        // Wrap in <p> if not already
        if (html.indexOf("<p>") !== 0) html = "<p>" + html + "</p>";
        return html;
    }

    function escapeHtml(text) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(text));
        return div.innerHTML;
    }

    function normalizeActionHref(href) {
        return typeof href === "string" && href.indexOf("/") === 0 && href.indexOf("//") !== 0 ? href : "#";
    }

    function flattenResponseText(response) {
        if (!response || typeof response !== "object") return "";
        return [response.message, response.followup]
            .filter(function (value) {
                return typeof value === "string" && value.trim();
            })
            .join("\n\n")
            .trim();
    }

    function recordHistoryTurn(role, content) {
        var text = String(content || "").trim();
        if (!text || ["user", "assistant"].indexOf(role) === -1) return;
        history.push({ role: role, content: text });
        if (history.length > 8) {
            history = history.slice(-8);
        }
    }

    function renderActionButtons(actions) {
        var buttons = actions
            .map(function (action) {
                var href = normalizeActionHref(action && action.href);
                var label = escapeHtml((action && action.label) || "Open");
                var style = action && action.style === "secondary" ? "secondary" : "primary";
                return (
                    '<a class="message-action message-action--' +
                    style +
                    '" href="' +
                    href +
                    '">' +
                    label +
                    "</a>"
                );
            })
            .join("");

        return '<div class="message-actions">' + buttons + "</div>";
    }

    function scrollToBottom(smooth, targetEl) {
        requestAnimationFrame(function () {
            if (!messagesEl) return;
            var behavior = smooth && !prefersReducedMotion() ? "smooth" : "auto";
            var scrollTarget = targetEl || messagesEl.lastElementChild;
            if (scrollTarget && typeof scrollTarget.scrollIntoView === "function") {
                scrollTarget.scrollIntoView({
                    block: "end",
                    behavior: behavior,
                });
                return;
            }

            window.scrollTo({
                top: document.documentElement.scrollHeight,
                behavior: behavior,
            });
        });
    }

    function autoResizeInput() {
        if (!inputEl || inputEl.tagName !== "TEXTAREA") return;
        inputEl.style.height = "0px";
        var minHeight = parseFloat(window.getComputedStyle(inputEl).minHeight) || 0;
        var nextHeight = Math.min(Math.max(inputEl.scrollHeight, minHeight), 220);
        inputEl.style.height = nextHeight + "px";
        updateComposerOffset();
    }

    // ------------------------------------------------------------------
    // Typing indicator
    // ------------------------------------------------------------------
    function showTyping() {
        typingEl.style.display = "flex";
        scrollToBottom(true, typingEl);
    }

    function hideTyping() {
        typingEl.style.display = "none";
    }

    // ------------------------------------------------------------------
    // Send message
    // ------------------------------------------------------------------
    async function sendMessage(text) {
        text = text.trim();
        if (!text || isSending) return;

        isSending = true;
        sendBtn.disabled = true;
        inputEl.value = "";

        // Hide suggestions after first user message
        hideSuggestions();

        // Show user bubble
        addMessage("user", text);
        showTyping();
        autoResizeInput();

        try {
            var res = await fetch("/api/chatbot", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: text,
                    history: history.slice(-8),
                }),
            });

            var data = await res.json();
            hideTyping();

            if (data.success && data.response) {
                recordHistoryTurn("user", text);
                recordHistoryTurn("assistant", flattenResponseText(data.response));
                addMessage("bot", data.response.message || "", data.response);
            } else {
                addMessage(
                    "bot",
                    data.fallback ||
                        data.error ||
                        "Something went wrong. Please try again.",
                    { type: "error", source: "error" }
                );
            }
        } catch (e) {
            hideTyping();
            addMessage(
                "bot",
                "Network error — please check your connection and try again.",
                { type: "error", source: "error" }
            );
        }

        isSending = false;
        sendBtn.disabled = false;
        inputEl.focus();
        autoResizeInput();
    }

    // ------------------------------------------------------------------
    // Event listeners
    // ------------------------------------------------------------------
    sendBtn.addEventListener("click", function () {
        sendMessage(inputEl.value);
    });

    if (fileButtonEl && fileInputEl) {
        fileButtonEl.addEventListener("click", function () {
            fileInputEl.click();
        });

        fileInputEl.addEventListener("change", function () {
            if (!fileInputEl.files || !fileInputEl.files.length) return;
            fileInputEl.value = "";
            inputEl.focus();
        });
    }

    inputEl.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage(inputEl.value);
        }
    });

    inputEl.addEventListener("input", autoResizeInput);

    // Accordion toggle (element may not exist on simplified pages)
    if (accordionBtn) {
        accordionBtn.addEventListener("click", function () {
            var icon = accordionBtn.querySelector(".toggle-icon");
            if (accordionBody.style.display === "none") {
                accordionBody.style.display = "block";
                accordionBtn.setAttribute("aria-expanded", "true");
                if (icon) icon.classList.add("open");
            } else {
                accordionBody.style.display = "none";
                accordionBtn.setAttribute("aria-expanded", "false");
                if (icon) icon.classList.remove("open");
            }
        });
    }

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------
    loadSuggestions();
    autoResizeInput();
    updateComposerOffset();

    if ("ResizeObserver" in window && composerEl) {
        var composerObserver = new ResizeObserver(function () {
            updateComposerOffset();
        });
        composerObserver.observe(composerEl);
    }

    window.addEventListener("resize", updateComposerOffset);

    inputEl.focus();
})();
