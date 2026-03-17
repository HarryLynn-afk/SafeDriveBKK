/* =============================================
   Bangkok Safety Assistant — Chat Logic
   ============================================= */

(function () {
    "use strict";

    // DOM refs
    const messagesEl = document.getElementById("chatMessages");
    const inputEl = document.getElementById("chatInput");
    const sendBtn = document.getElementById("chatSend");
    const suggestionsEl = document.getElementById("chatSuggestions");
    const sugButtonsEl = document.getElementById("suggestionsButtons");
    const typingEl = document.getElementById("typingIndicator");
    const accordionBtn = document.getElementById("howItWorksToggle");
    const accordionBody = document.getElementById("howItWorksContent");

    let isSending = false;

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
                "Where should Bangkok install speed cameras?",
            ]);
        }
    }

    function renderSuggestions(items) {
        sugButtonsEl.innerHTML = "";
        items.forEach(function (text) {
            var btn = document.createElement("button");
            btn.className = "suggestion-btn";
            btn.textContent = text;
            btn.addEventListener("click", function () {
                sendMessage(text);
            });
            sugButtonsEl.appendChild(btn);
        });
    }

    // ------------------------------------------------------------------
    // Messages
    // ------------------------------------------------------------------
    function addMessage(role, content, meta) {
        var wrapper = document.createElement("div");
        wrapper.className = "message " + role;

        var avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.textContent = role === "bot" ? "\uD83E\uDD16" : "\uD83D\uDC64";

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
        scrollToBottom();
    }

    function formatContent(content, meta) {
        if (!meta) return escapeHtml(content);

        // Error type
        if (meta.type === "error") {
            return "<p>" + escapeHtml(meta.message || content) + "</p>";
        }

        // AI response — may contain markdown-style formatting
        if (meta.type === "ai" || meta.source === "groq") {
            return formatMarkdown(meta.message || content);
        }

        // Scripted responses
        if (meta.type === "list" && meta.items) {
            var header = meta.message ? "<p>" + escapeHtml(meta.message) + "</p>" : "";
            var items = meta.items
                .map(function (item) {
                    return "<li>" + escapeHtml(item) + "</li>";
                })
                .join("");
            var footer = meta.follow_up
                ? "<p><em>" + escapeHtml(meta.follow_up) + "</em></p>"
                : "";
            return header + "<ol>" + items + "</ol>" + footer;
        }

        if (meta.type === "stats" && meta.stats) {
            var header = meta.message ? "<p>" + escapeHtml(meta.message) + "</p>" : "";
            var rows = Object.keys(meta.stats)
                .map(function (k) {
                    return (
                        "<li><strong>" +
                        escapeHtml(k) +
                        ":</strong> " +
                        escapeHtml(String(meta.stats[k])) +
                        "</li>"
                    );
                })
                .join("");
            var footer = meta.follow_up
                ? "<p><em>" + escapeHtml(meta.follow_up) + "</em></p>"
                : "";
            return header + "<ul>" + rows + "</ul>" + footer;
        }

        if (meta.type === "zone_detail") {
            var parts = [];
            if (meta.message) parts.push("<p>" + escapeHtml(meta.message) + "</p>");
            if (meta.details) {
                var dRows = Object.keys(meta.details)
                    .map(function (k) {
                        return (
                            "<li><strong>" +
                            escapeHtml(k) +
                            ":</strong> " +
                            escapeHtml(String(meta.details[k])) +
                            "</li>"
                        );
                    })
                    .join("");
                parts.push("<ul>" + dRows + "</ul>");
            }
            if (meta.follow_up)
                parts.push("<p><em>" + escapeHtml(meta.follow_up) + "</em></p>");
            return parts.join("");
        }

        // Default: plain text with follow_up
        var out = "<p>" + escapeHtml(meta.message || content) + "</p>";
        if (meta.follow_up)
            out += "<p><em>" + escapeHtml(meta.follow_up) + "</em></p>";
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

    function scrollToBottom() {
        requestAnimationFrame(function () {
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });
    }

    // ------------------------------------------------------------------
    // Typing indicator
    // ------------------------------------------------------------------
    function showTyping() {
        typingEl.style.display = "flex";
        scrollToBottom();
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
        suggestionsEl.style.display = "none";

        // Show user bubble
        addMessage("user", text);
        showTyping();

        try {
            var res = await fetch("/api/chatbot", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text }),
            });

            var data = await res.json();
            hideTyping();

            if (data.success && data.response) {
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
    }

    // ------------------------------------------------------------------
    // Event listeners
    // ------------------------------------------------------------------
    sendBtn.addEventListener("click", function () {
        sendMessage(inputEl.value);
    });

    inputEl.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage(inputEl.value);
        }
    });

    // Accordion toggle
    accordionBtn.addEventListener("click", function () {
        var icon = accordionBtn.querySelector(".toggle-icon");
        if (accordionBody.style.display === "none") {
            accordionBody.style.display = "block";
            icon.classList.add("open");
        } else {
            accordionBody.style.display = "none";
            icon.classList.remove("open");
        }
    });

    // ------------------------------------------------------------------
    // Init
    // ------------------------------------------------------------------
    loadSuggestions();
    inputEl.focus();
})();
