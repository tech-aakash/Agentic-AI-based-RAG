// ✅ Load marked.js in your HTML (before this script runs):
// <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

const form = document.getElementById("chat-form");
const chatBox = document.getElementById("chat-box");

// Optional: Basic sanitizer to avoid script injection (very light)
function sanitizeHTML(html) {
  const temp = document.createElement("div");
  temp.textContent = html;
  return temp.innerHTML;
}

// Function to render Markdown safely
function renderMarkdown(markdownText) {
  // Configure marked if needed
  marked.setOptions({
    breaks: true, // Line breaks
    gfm: true, // GitHub-flavored markdown
  });

  // Convert markdown → HTML
  let html = marked.parse(markdownText || "");

  // Optional: sanitize final HTML if you're concerned about security
  // (use DOMPurify if you want stronger sanitization)
  return html;
}

// Handle chat form submission
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const questionInput = document.getElementById("question");
  const question = questionInput.value.trim();
  if (!question) return;

  // 🧍 User message
  chatBox.innerHTML += `
    <div class="message user">
      <div class="bubble">${sanitizeHTML(question)}</div>
      <img src="/static/assets/user.png" class="avatar" alt="User">
    </div>`;
  chatBox.scrollTop = chatBox.scrollHeight;
  questionInput.value = "";

  // 💭 Typing (thinking) animation
  const typingDiv = document.createElement("div");
  typingDiv.classList.add("message", "bot");
  typingDiv.innerHTML = `
    <img src="/static/assets/bot.png" class="avatar" alt="Bot">
    <div class="bubble typing">thinking<span>.</span><span>.</span><span>.</span></div>
  `;
  chatBox.appendChild(typingDiv);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ question }),
    });
    const data = await res.json();

    // Remove typing animation
    typingDiv.remove();

    // Render markdown in bot response
    const formattedResponse = renderMarkdown(data.response || "⚠️ Something went wrong.");

    // 🧠 Bot response
    const botMessage = document.createElement("div");
    botMessage.classList.add("message", "bot");
    botMessage.innerHTML = `
      <img src="/static/assets/bot.png" class="avatar" alt="Bot">
      <div class="bubble markdown">${formattedResponse}</div>
    `;
    chatBox.appendChild(botMessage);

  } catch (error) {
    typingDiv.remove();
    chatBox.innerHTML += `
      <div class="message bot">
        <img src="/static/assets/bot.png" class="avatar" alt="Bot">
        <div class="bubble">⚠️ Network error. Please try again.</div>
      </div>`;
  }

  chatBox.scrollTop = chatBox.scrollHeight;
});