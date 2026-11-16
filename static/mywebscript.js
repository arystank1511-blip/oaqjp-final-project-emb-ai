const form = document.getElementById('emotionForm');
const input = document.getElementById('textToAnalyze');
const output = document.getElementById('system_response');
const button = form.querySelector('button');

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const query = new URLSearchParams({textToAnalyze: input.value});
    button.disabled = true;
    output.textContent = 'Analyzing…';

    try {
        const response = await fetch(`/emotionDetector?${query}`);
        const message = await response.text();
        const match = response.ok && /^(.*The dominant emotion is )(\w+)\.$/.exec(message);
        if (match) {
            const name = document.createElement('strong');
            name.textContent = match[2];
            output.replaceChildren(match[1], name, '.');
        } else {
            output.textContent = message;
        }
    } catch {
        output.textContent = 'Could not reach the server. Please try again.';
    } finally {
        button.disabled = false;
    }
});
