function predictHappiness() {
    const form = document.getElementById('predictionForm');
    const formData = new FormData(form);
    const data = {};

    formData.forEach((value, key) => {
        data[key] = value;
    });

    // 按钮 loading 状态
    const btn = document.querySelector('.btn-primary');
    const originalText = btn.innerHTML;
    btn.innerHTML = '⏳ 计算中...';
    btn.disabled = true;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(result => {
            if (result.error) {
                alert('预测出错: ' + result.error);
                return;
            }
            showResult(result);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('网络错误，请重试');
        })
        .finally(() => {
            btn.innerHTML = originalText;
            btn.disabled = false;
        });
}

function showResult(result) {
    document.getElementById('emptyResult').style.display = 'none';
    const resultDiv = document.getElementById('predictionResult');
    resultDiv.style.display = 'flex'; // Restore flex display

    // Update Score with Animation
    const scoreElement = document.getElementById('resultScore');
    const targetScore = result.happiness_score || 0;
    let currentScore = 0;
    const duration = 1500; // ms
    const startTime = performance.now();

    function animateScore(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out quart
        const ease = 1 - Math.pow(1 - progress, 4);

        currentScore = targetScore * ease;
        scoreElement.innerText = currentScore.toFixed(1);

        if (progress < 1) {
            requestAnimationFrame(animateScore);
        }
    }
    requestAnimationFrame(animateScore);

    // Update Secondary Info
    const emojis = { 1: '😭', 2: '😟', 3: '😐', 4: '🙂', 5: '😍' };
    const levelText = result.message.split(' - ')[0]; // e.g. "幸福"

    document.getElementById('resultEmoji').innerText = emojis[result.happiness_level] || '❓';
    document.getElementById('resultLevel').innerText = `${levelText} (Level ${result.happiness_level})`;
    document.getElementById('resultMessage').innerText = result.message.split(' - ')[1] || '';

    // Update Probabilities
    for (let i = 1; i <= 5; i++) {
        const probStr = result.probability[i]; // "XX.X%"
        document.getElementById(`prob${i}`).style.width = probStr;
        document.getElementById(`prob${i}Text`).innerText = probStr;
    }
}


function randomFill() {
    // 预设的人画像 (5个等级)
    const personas = [
        {   // Level 1: 极度不幸 (最低配)
            age: 50, gender: 1, edu: 1, political: 4,
            income: 0, family_income: 0,
            house: 0, car: 0,
            equity: 1, depression: 5, class: 1, status_peer: 1,
            province: 24, city: 1, county: 1, hukou: 1
        },
        {   // Level 2: 不幸福 (低收入/高压)
            age: 35, gender: 2, edu: 8, political: 4,
            income: 20000, family_income: 30000,
            house: 0, car: 0,
            equity: 2, depression: 4, class: 2, status_peer: 1,
            province: 15, city: 1, county: 1, hukou: 1
        },
        {   // Level 3: 一般 (普通)
            age: 28, gender: 1, edu: 12, political: 3,
            income: 70000, family_income: 120000,
            house: 1, car: 0,
            equity: 3, depression: 3, class: 4, status_peer: 2,
            province: 11, city: 1, county: 1, hukou: 2
        },
        {   // Level 4: 幸福 (优渥)
            age: 32, gender: 2, edu: 13, political: 1,
            income: 200000, family_income: 400000,
            house: 1, car: 1,
            equity: 4, depression: 2, class: 7, status_peer: 2,
            province: 31, city: 1, county: 1, hukou: 2
        },
        {   // Level 5: 完美人生 (顶配)
            age: 45, gender: 1, edu: 14, political: 1,
            income: 2000000, family_income: 5000000,
            house: 5, car: 3,
            equity: 5, depression: 1, class: 10, status_peer: 3,
            province: 11, city: 1, county: 1, hukou: 3
        }
    ];

    // 随机选择一个画像
    const randomPersona = personas[Math.floor(Math.random() * personas.length)];

    // 填充表单
    const setInputValue = (name, value) => {
        const input = document.querySelector(`[name="${name}"]`);
        if (input) {
            input.value = value;
            // 触发 input 事件以确保任何监听器都被通知
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
        }
    };

    // 遍历画像属性并填充
    for (const [key, value] of Object.entries(randomPersona)) {
        setInputValue(key, value);
    }

    // 添加一点随机波动，避免每次完全一样
    const incomeInput = document.querySelector('[name="income"]');
    if (incomeInput) {
        const baseIncome = parseInt(incomeInput.value);
        incomeInput.value = baseIncome + Math.floor(Math.random() * 5000) - 2500;
    }
}

