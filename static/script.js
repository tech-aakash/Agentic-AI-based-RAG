const form = document.getElementById("chat-form");
const chatBox = document.getElementById("chat-box");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const questionInput = document.getElementById("question");
  const question = questionInput.value.trim();
  if (!question) return;

  // 🧍 User message
  chatBox.innerHTML += `
    <div class="message user">
      <div class="bubble">${question}</div>
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

    // 🧠 Bot response
    chatBox.innerHTML += `
      <div class="message bot">
        <img src="/static/assets/bot.png" class="avatar" alt="Bot">
        <div class="bubble">${data.response || "⚠️ Something went wrong."}</div>
      </div>`;
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
