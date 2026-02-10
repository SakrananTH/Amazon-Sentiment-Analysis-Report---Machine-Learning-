document.addEventListener('DOMContentLoaded', () => {
    fetchData();

    const analyzeBtn = document.getElementById('analyze-btn');
    const reviewInput = document.getElementById('review-input');
    const exampleChips = document.querySelectorAll('.chip');

    // Handle Quick Examples
    exampleChips.forEach(chip => {
        chip.addEventListener('click', () => {
            reviewInput.value = chip.dataset.text;
            resultSection.style.display = 'none'; // Hide previous results
        });
    });

    // Handle Analyze Button
    analyzeBtn.addEventListener('click', async () => {
        const text = reviewInput.value.trim();
        const errorMsg = document.getElementById('error-msg');
        
        if (!text) {
            showError('กรุณาป้อนข้อความก่อน (Please enter text)');
            return;
        }
        
        showError(''); // Clear error
        setLoading(true);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            displayResult(data);
        } catch (error) {
            console.error('Error:', error);
            showError('เกิดข้อผิดพลาดในการวิเคราะห์ (Analysis Error)');
        } finally {
            setLoading(false);
        }
    });
});

const resultSection = document.getElementById('result-section');

function displayResult(data) {
    const sentimentText = document.getElementById('sentiment-text');
    const sentimentIcon = document.getElementById('sentiment-icon');
    const confidenceVal = document.getElementById('confidence-val');
    const latencyVal = document.getElementById('latency-val');
    
    // Show section
    resultSection.style.display = 'block';
    
    // Set values
    confidenceVal.textContent = (data.confidence * 100).toFixed(1) + '%';
    latencyVal.textContent = data.latency.toFixed(2) + ' ms';
    
    if (data.sentiment === 'Positive') {
        sentimentText.textContent = 'เชิงบวก (Positive)';
        sentimentText.style.color = 'var(--success-color)';
        sentimentIcon.innerHTML = '😄';
    } else {
        sentimentText.textContent = 'เชิงลบ (Negative)';
        sentimentText.style.color = 'var(--error-color)';
        sentimentIcon.innerHTML = '😡';
    }
}

function showError(msg) {
    const el = document.getElementById('error-msg');
    el.textContent = msg;
    el.style.display = msg ? 'block' : 'none';
}

function setLoading(isLoading) {
    const btn = document.getElementById('analyze-btn');
    if (isLoading) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> กำลังวิเคราะห์...';
    } else {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-search"></i> วิเคราะห์ความรู้สึก';
    }
}

async function fetchData() {
    try {
        const response = await fetch('/api/data');
        const result = await response.json();
        
        if (result.error) return;

        // Update Total Count
        document.getElementById('total-count').textContent = result.total_count.toLocaleString();

        // Populate Table
        const tbody = document.querySelector('#data-table tbody');
        tbody.innerHTML = '';
        
        result.data.forEach(row => {
            const tr = document.createElement('tr');
            // Truncate text for table
            const shortText = row.text.length > 50 ? row.text.substring(0, 50) + '...' : row.text;
            
            tr.innerHTML = `
                <td title="${row.text}">${shortText}</td>
                <td><span class="badge ${row.label.toLowerCase()}">${row.label}</span></td>
            `;
            tbody.appendChild(tr);
        });
        
    } catch (e) {
        console.error("Failed to fetch dataset info", e);
    }
}
