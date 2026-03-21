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

    getAvatarSvg(role) {
        if (role === 'user') {
            return `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M15.4 8.6a3.4 3.4 0 1 1-6.8 0 3.4 3.4 0 0 1 6.8 0Z"></path>
                    <path d="M5.6 18.2c1.55-2.4 3.74-3.65 6.4-3.65s4.85 1.25 6.4 3.65"></path>
                </svg>
            `;
        }

        return `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 3.25 18 5.75v5.6c0 4.2-2.7 7.97-6 9.4-3.3-1.43-6-5.2-6-9.4v-5.6L12 3.25Z"></path>
                <path d="m9.2 12.1 1.8 1.8 3.8-4"></path>
            </svg>
        `;
    }

    createAvatar(role) {
        const avatar = document.createElement('div');
        avatar.className = `message-avatar message-avatar--${role}`;
        avatar.setAttribute('aria-hidden', 'true');
        avatar.innerHTML = this.getAvatarSvg(role);
        return avatar;
    }

    createMessageShell(role, bubbleHtml) {
        const div = document.createElement('div');
        div.className = `message ${role}`;

        const content = document.createElement('div');
        content.className = 'message-content';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = bubbleHtml;

        content.appendChild(bubble);
        div.appendChild(this.createAvatar(role));
        div.appendChild(content);
        return div;
    }

    init() {
        this.syncFabState();
        this.fab.addEventListener('click', () => this.toggle());
        this.closeBtn.addEventListener('click', () => this.close());

        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        // Set context-aware suggestions based on current page
        this.setContextualSuggestions();
    }

    syncFabState() {
        const label = this.isOpen ? 'Close Safety Assistant' : 'Open Safety Assistant';
        this.fab.classList.toggle('active', this.isOpen);
        this.fab.setAttribute('aria-expanded', String(this.isOpen));
        this.fab.setAttribute('aria-label', label);
        this.fab.setAttribute('title', label);
        this.widget.classList.toggle('active', this.isOpen);
        this.widget.setAttribute('aria-hidden', String(!this.isOpen));
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
        this.isOpen = true;
        this.syncFabState();
        setTimeout(() => this.input.focus(), 100);
    }

    close() {
        this.isOpen = false;
        this.syncFabState();
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
        const bodyHtml = `<div class="message-body">${this.renderRichText(text)}</div>`;
        const div = this.createMessageShell(role, bodyHtml);
        this.messages.appendChild(div);
        this.scrollToBottom();
    }

    addBotResponse(response) {
        let html = `<div class="message-body">${this.renderRichText(response.message || '')}</div>`;

        if (response.followup) {
            html += `<div class="message-followup">${this.renderRichText(response.followup)}</div>`;
        }

        const div = this.createMessageShell('bot', html);
        this.messages.appendChild(div);
        this.scrollToBottom();
    }

    renderRichText(text) {
        const blocks = [];
        const lines = String(text || '').split('\n');
        let listType = null;
        let listItems = [];

        const flushList = () => {
            if (!listItems.length || !listType) return;
            const tag = listType === 'ordered' ? 'ol' : 'ul';
            const items = listItems
                .map(item => `<li>${this.formatInline(item)}</li>`)
                .join('');
            blocks.push(`<${tag} class="message-list">${items}</${tag}>`);
            listItems = [];
            listType = null;
        };

        lines.forEach((rawLine) => {
            const line = rawLine.trim();
            if (!line) {
                flushList();
                return;
            }

            const bulletMatch = line.match(/^[•*-]\s+(.+)$/);
            if (bulletMatch) {
                if (listType && listType !== 'unordered') {
                    flushList();
                }
                listType = 'unordered';
                listItems.push(bulletMatch[1]);
                return;
            }

            const orderedMatch = line.match(/^\d+[.)]\s+(.+)$/);
            if (orderedMatch) {
                if (listType && listType !== 'ordered') {
                    flushList();
                }
                listType = 'ordered';
                listItems.push(orderedMatch[1]);
                return;
            }

            flushList();
            blocks.push(`<p>${this.formatInline(line)}</p>`);
        });

        flushList();
        return blocks.join('');
    }

    formatInline(text) {
        let safe = this.escapeHtml(text);
        safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        return safe;
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
        div.appendChild(this.createAvatar('bot'));

        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = `
            <div class="typing-indicator-widget">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        div.appendChild(content);
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
