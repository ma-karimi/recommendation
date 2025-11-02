# راهنمای کامل API سیستم توصیه محصولات

تاریخ: 2024-11-01

---

## 🚀 راه‌اندازی API

### 1. نصب پیش‌نیازها

```bash
# نصب Python dependencies
pip install -r requirements.txt

# نصب و راه‌اندازی Redis
# macOS
brew install redis
brew services start redis

# Linux
sudo apt install redis-server
sudo systemctl start redis
```

### 2. تولید توصیه‌ها و ذخیره در Redis

```bash
# تولید توصیه برای همه کاربران
python generate_recommendations.py --all

# یا برای تست (1000 کاربر)
python generate_recommendations.py --sample 1000
```

### 3. راه‌اندازی API سرور

```bash
# روش 1: استفاده از run_recommendation.py
python run_recommendation.py api --host 0.0.0.0 --port 8000

# روش 2: مستقیم با uvicorn
uvicorn recommendation_api:app --host 0.0.0.0 --port 8000

# روش 3: با reload در حالت development
uvicorn recommendation_api:app --reload --host 0.0.0.0 --port 8000
```

### 4. بررسی سلامت سیستم

```bash
curl http://localhost:8000/health

# پاسخ:
{
  "status": "healthy",
  "recommender_ready": false,
  "products_loaded": 36114
}
```

---

## 📡 Endpoints موجود

### 1. `/` - صفحه اصلی

```bash
GET http://localhost:8000/

# پاسخ:
{
  "message": "سیستم توصیه محصولات",
  "version": "1.0.0",
  "status": "active"
}
```

---

### 2. `/health` - بررسی سلامت ✅

```bash
GET http://localhost:8000/health

# پاسخ:
{
  "status": "healthy",
  "recommender_ready": false,
  "products_loaded": 36114
}
```

**دلایل استفاده:**
- بررسی اینکه API در حال اجرا است
- کنترل سلامت کلی
- Monitoring و Alerting

---

### 3. `/recommendations/{user_id}` - دریافت توصیه‌ها ⭐

**این endpoint اصلی API شماست!**

```bash
GET http://localhost:8000/recommendations/9194445?limit=10

# پاسخ نمونه:
[
  {
    "product_id": 501838,
    "score": 1587.0,
    "confidence": 1.0,
    "reason": "Collaborative: 1 کاربران مشابه این محصول را خریده‌اند",
    "product_title": null,
    "product_price": null,
    "product_stock": null,
    "collaborative_details": {
      "similar_users": [
        {
          "user_id": 9391201,
          "similarity": 0.0009,
          "similarity_percent": 0.09
        }
      ],
      "total_similar_users": 1
    }
  }
]
```

**پارامترها:**
- `user_id` (path): شناسه کاربر (required)
- `limit` (query): تعداد توصیه‌ها (default: 10)
- `use_redis` (query): استفاده از Redis (default: true)

**مثال‌ها:**

```bash
# دریافت 5 توصیه اول
curl "http://localhost:8000/recommendations/9194445?limit=5"

# دریافت همه توصیه‌ها (20 مورد)
curl "http://localhost:8000/recommendations/9194445?limit=20"

# بدون استفاده از Redis (fallback)
curl "http://localhost:8000/recommendations/9194445?use_redis=false"
```

---

### 4. `/stats` - آمار سیستم

```bash
GET http://localhost:8000/stats

# پاسخ:
{
  "total_products": 36114,
  "recommender_ready": false,
  "redis_connected": true,
  "redis_stats": {
    "total_recommendations": 3,
    "memory_usage_mb": 80.27
  }
}
```

---

### 5. `/insights/{user_id}` - بینش‌های کاربر

```bash
GET http://localhost:8000/insights/9194445

# پاسخ:
{
  "total_interactions": 10,
  "preferred_categories": [...],
  "average_purchase_value": 150000.0,
  "similar_users": [...]
}
```

---

### 6. `/popular` - محصولات محبوب

```bash
GET http://localhost:8000/popular?limit=10

# پاسخ:
[
  {
    "product_id": 12345,
    "purchase_count": 150,
    "product_title": "محصول نمونه",
    "product_price": 100000.0
  }
]
```

---

### 7. `/similar/{product_id}` - محصولات مشابه

```bash
GET http://localhost:8000/similar/12345?limit=5

# پاسخ:
[
  {
    "product_id": 67890,
    "similarity_score": 0.85,
    "product_title": "محصول مشابه",
    "product_price": 95000.0
  }
]
```

---

### 8. `/retrain` - بازآموزی مدل

```bash
POST http://localhost:8000/retrain

# پاسخ:
{
  "message": "مدل با موفقیت بازآموزی شد",
  "products_count": 36114
}
```

**توجه:** این endpoint می‌تواند 10-45 دقیقه طول بکشد!

---

## 🔗 مثال‌های استفاده در زبان‌های مختلف

### Python

```python
import requests

# دریافت توصیه‌ها
response = requests.get("http://localhost:8000/recommendations/9194445?limit=10")
recommendations = response.json()

for rec in recommendations:
    print(f"Product: {rec['product_id']}")
    print(f"Score: {rec['score']}")
    print(f"Reason: {rec['reason']}")
    
    # نمایش جزئیات Collaborative
    if rec.get('collaborative_details'):
        details = rec['collaborative_details']
        print(f"Similar Users: {details['total_similar_users']}")
```

### JavaScript / Node.js

```javascript
// با fetch
const response = await fetch('http://localhost:8000/recommendations/9194445?limit=10');
const recommendations = await response.json();

recommendations.forEach(rec => {
    console.log(`Product: ${rec.product_id}`);
    console.log(`Score: ${rec.score}`);
    
    if (rec.collaborative_details) {
        console.log(`Similar Users: ${rec.collaborative_details.total_similar_users}`);
    }
});

// با axios
const axios = require('axios');
const recommendations = await axios.get('http://localhost:8000/recommendations/9194445');
console.log(recommendations.data);
```

### PHP / Laravel

```php
<?php

use Illuminate\Support\Facades\Http;

// دریافت توصیه‌ها
$response = Http::get('http://localhost:8000/recommendations/9194445', [
    'limit' => 10
]);

$recommendations = $response->json();

foreach ($recommendations as $rec) {
    echo "Product: {$rec['product_id']}\n";
    echo "Score: {$rec['score']}\n";
    
    if (isset($rec['collaborative_details'])) {
        $similarUsers = $rec['collaborative_details']['total_similar_users'];
        echo "Similar Users: {$similarUsers}\n";
    }
}
```

### cURL

```bash
# دریافت 10 توصیه
curl -X GET "http://localhost:8000/recommendations/9194445?limit=10" \
  -H "Accept: application/json"

# با jq برای format زیبا
curl -s http://localhost:8000/recommendations/9194445 | jq '.[] | {product_id, score, reason}'

# بررسی health
curl http://localhost:8000/health

# آمار سیستم
curl http://localhost:8000/stats
```

---

## 🔒 امنیت و Authentication

برای استفاده در Production، اضافه کردن Authentication توصیه می‌شود:

### 1. API Key Authentication

```python
# در recommendation_api.py
from fastapi import Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    api_key = credentials.credentials
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# استفاده در endpoint
@app.get("/recommendations/{user_id}")
async def get_user_recommendations(
    user_id: int,
    limit: int = 10,
    api_key = Depends(verify_api_key)
):
    # ...
```

**استفاده:**
```bash
curl -H "Authorization: Bearer your-api-key" \
  http://localhost:8000/recommendations/9194445
```

---

## 📊 Performance

### سرعت endpoint

```
/recommendations/{user_id}:
- Redis Read: < 1ms ✅
- Fallback (direct): ~50-100ms ⚠️

/batch (future):
- Multiple reads: ~5-10ms ✅
```

### Load Testing

```bash
# با Apache Bench
ab -n 1000 -c 10 http://localhost:8000/recommendations/9194445

# با wrk
wrk -t4 -c100 -d30s http://localhost:8000/recommendations/9194445
```

### بهینه‌سازی

1. **Connection Pooling:**
   ```python
   # استفاده از Redis connection pool
   # در recommendation_storage.py فعال است
   ```

2. **Caching:**
   ```python
   # محصولات در memory کش می‌شوند
   products_cache = {p.id: p for p in products}
   ```

3. **Async/Await:**
   - همه endpoints از async استفاده می‌کنند
   - uvicorn با workers برای scalability

---

## 🌐 Deploy در Production

### 1. با Gunicorn

```bash
gunicorn recommendation_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### 2. با Docker

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "recommendation_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build & Run
docker build -t recommendation-api .
docker run -p 8000:8000 --network host recommendation-api
```

### 3. با Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

```bash
docker-compose up -d
```

### 4. با Nginx (Reverse Proxy)

```nginx
upstream recommendation_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://recommendation_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📖 مستندات Swagger

دسترسی به مستندات تعاملی:

```
http://localhost:8000/docs
```

این صفحه شامل:
- لیست همه endpoints
- مثال‌های request/response
- امکان تست مستقیم در مرورگر

---

## 🔍 Monitoring و Logging

### مشاهده Logs

```bash
# با uvicorn
uvicorn recommendation_api:app --log-level debug

# مشاهده real-time
tail -f logs/api.log
```

### Health Check Script

```bash
#!/bin/bash
# health_check.sh

API_URL="http://localhost:8000/health"
STATUS=$(curl -s $API_URL | jq -r '.status')

if [ "$STATUS" = "healthy" ]; then
    echo "✅ API is healthy"
    exit 0
else
    echo "❌ API is not healthy"
    exit 1
fi
```

### Prometheus Metrics (Future)

```python
# اضافه کردن prometheus metrics
from prometheus_client import Counter, Histogram

request_count = Counter('http_requests_total', 'Total HTTP requests')
request_duration = Histogram('http_request_duration_seconds', 'Request duration')
```

---

## ❓ Troubleshooting

### مشکل: "503 Service Unavailable"

**دلایل:**
- Redis وصل نیست
- توصیه‌ها تولید نشده‌اند

**راه حل:**
```bash
# بررسی Redis
redis-cli ping  # باید PONG برگرداند

# تولید توصیه‌ها
python generate_recommendations.py --all
```

### مشکل: "500 Internal Server Error"

**دلایل:**
- خطا در کد
- دیتابیس وصل نیست

**راه حل:**
```bash
# بررسی logs
tail -f logs/error.log

# تست اتصال دیتابیس
python test_db_connection.py
```

### مشکل: عملکرد کند

**دلایل:**
- استفاده از fallback به جای Redis
- عدم وجود cache

**راه حل:**
```bash
# بررسی Redis stats
curl http://localhost:8000/stats

# Clear cache و reload
docker-compose restart api
```

---

## 📞 تماس و پشتیبانی

- **مستندات:** `README.md`
- **مثال‌ها:** `examples_usage.py`
- **نیازمندی‌های منابع:** `RESOURCE_REQUIREMENTS.md`

---

**موفق باشید! 🚀**

