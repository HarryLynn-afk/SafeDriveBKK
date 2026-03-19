// Floating Chat Widget
class FloatingChat {
    constructor() {
        this.fab = document.getElementById('chatFab');
        this.widget = document.getElementById('chatWidget');
        this.closeBtn = document.getElementById('chatWidgetClose');
        this.input = document.getElementById('chatWidgetInput');
        this.sendBtn = document.getElementById('chatWidgetSend');
        this.messages = document.getElementById('chatWidgetMessages');
        this.suggestions = document.getElementById('chatWidgetSuggestions');

        if (!this.fab || !this.widget) return;

        this.isOpen = false;
        this.init();
    }

    init() {
        this.fab.addEventListener('click', () => this.toggle());
        this.closeBtn.addEventListener('click', () => this.close());

        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        // Set context-aware suggestions based on current page
        this.setContextualSuggestions();
    }

    setContextualSuggestions() {
        const path = window.location.pathname;

        let suggestions = [];

        if (path.includes('trip-planner') || path.includes('trip_planner')) {
            suggestions = [
                { text: 'Best time to travel?', question: 'What is the best time to travel in Bangkok?' },
                { text: 'Safer route alternatives?', question: 'Show me safer route alternatives' },
                { text: 'Avoid dangerous zones?', question: 'Which zones should I avoid?' }
            ];
        } else if (path.includes('analytics')) {
            suggestions = [
                { text: 'Peak accident times?', question: 'When do most accidents happen?' },
                { text: 'Main accident causes?', question: 'What are the main causes of accidents?' },
                { text: 'Weekend vs weekday?', question: 'Are weekends more dangerous than weekdays?' }
            ];
        } else if (path.includes('about')) {
            suggestions = [
                { text: 'How does it work?', question: 'How does this system work?' },
                { text: 'What data is used?', question: 'What data do you use?' },
                { text: 'Top recommendations?', question: 'What should Bangkok do to improve safety?' }
            ];
        } else {
            suggestions = [
                { text: 'Most dangerous areas?', question: 'Which areas are most dangerous?' },
                { text: 'Best time to drive?', question: 'What is the best time to drive?' },
                { text: 'Is Lat Krabang safe?', question: 'Is Lat Krabang safe?' }
            ];
        }

        this.suggestions.innerHTML = '';
        suggestions.forEach(sug => {
            const chip = document.createElement('button');
            chip.className = 'suggestion-chip';
            chip.textContent = sug.text;
            chip.dataset.question = sug.question;
            chip.addEventListener('click', () => this.sendMessage(sug.question));
            this.suggestions.appendChild(chip);
        });
    }

    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }

    open() {
        this.widget.classList.add('active');
        this.fab.classList.add('active');
        this.fab.innerHTML = '&#x2715;';
        this.isOpen = true;
        setTimeout(() => this.input.focus(), 100);
    }

    close() {
        this.widget.classList.remove('active');
        this.fab.classList.remove('active');
        this.fab.innerHTML = '&#x1F4AC;';
        this.isOpen = false;
    }

    async sendMessage(text) {
        const message = text || this.input.value.trim();
        if (!message) return;

        this.input.value = '';
        this.addMessage('user', message);

        // Hide suggestions after first message
        this.suggestions.style.display = 'none';

        this.showTyping();
        this.sendBtn.disabled = true;

        try {
            const response = await fetch('/api/chatbot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });

            const data = await response.json();
            this.hideTyping();
            this.sendBtn.disabled = false;

            if (data.success && data.response) {
                this.addBotResponse(data.response);
            } else {
                this.addMessage('bot', 'Sorry, something went wrong. Please try again.');
            }
        } catch (err) {
            this.hideTyping();
            this.sendBtn.disabled = false;
            this.addMessage('bot', 'Connection error. Please try again.');
        }
    }

    addMessage(role, text) {
        const div = document.createElement('div');
        div.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = role === 'bot' ? '&#x1F916;' : '&#x1F464;';

        const content = document.createElement('div');
        content.className = 'message-content';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = `<p>${this.escapeHtml(text)}</p>`;

        content.appendChild(bubble);
        div.appendChild(avatar);
        div.appendChild(content);
        this.messages.appendChild(div);
        this.scrollToBottom();
    }

    addBotResponse(response) {
        const div = document.createElement('div');
        div.className = 'message bot';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '&#x1F916;';

        const content = document.createElement('div');
        content.className = 'message-content';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        // Format the message - convert markdown-style bold and newlines
        let html = this.formatText(response.message || '');

        // Add followup if present
        if (response.followup) {
            html += `<p style="margin-top:8px;opacity:0.8;font-size:13px;">${this.formatText(response.followup)}</p>`;
        }

        bubble.innerHTML = html;
        content.appendChild(bubble);
        div.appendChild(avatar);
        div.appendChild(content);
        this.messages.appendChild(div);
        this.scrollToBottom();
    }

    formatText(text) {
        // Escape HTML first, then apply formatting
        let safe = this.escapeHtml(text);
        // Bold: **text**
        safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        // Newlines to <br> / paragraphs
        safe = safe.replace(/\n\n/g, '</p><p>');
        safe = safe.replace(/\n/g, '<br>');
        return `<p>${safe}</p>`;
    }

    escapeHtml(text) {
        const el = document.createElement('span');
        el.textContent = text;
        return el.innerHTML;
    }

    showTyping() {
        const div = document.createElement('div');
        div.className = 'message bot';
        div.id = 'widgetTyping';
        div.innerHTML = `
            <div class="message-avatar">&#x1F916;</div>
            <div class="message-content">
                <div class="typing-indicator-widget">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        this.messages.appendChild(div);
        this.scrollToBottom();
    }

    hideTyping() {
        const el = document.getElementById('widgetTyping');
        if (el) el.remove();
    }

    scrollToBottom() {
        this.messages.scrollTop = this.messages.scrollHeight;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.floatingChat = new FloatingChat();
});
