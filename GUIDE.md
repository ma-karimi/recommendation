# راهنمای کامل سیستم توصیه محصولات

<div dir="rtl">

راهنمای جامع برای استفاده از سیستم توصیه محصولات، شامل تمام جزئیات، مثال‌ها، تنظیمات و بهترین روش‌ها.

</div>

---

## 📚 فهرست مطالب

- [خلاصه پروژه](#خلاصه-پروژه)
- [نحوه استفاده](#نحوه-استفاده)
- [راه‌اندازی API](#راه‌اندازی-api)
- [API Endpoints](#api-endpoints)
- [استفاده در زبان‌های مختلف](#استفاده-در-زبان‌های-مختلف)
- [ساختار پروژه](#ساختار-پروژه)
- [نیازمندی‌های منابع](#نیازمندی‌های-منابع)
- [بهینه‌سازی و Performance](#بهینه‌سازی-و-performance)
- [Deploy در Production](#deploy-در-production)
- [Troubleshooting](#troubleshooting)
- [استفاده در Laravel](#استفاده-در-laravel)

---

## 📋 خلاصه پروژه

### سیستم توصیه ترکیبی (Hybrid Recommender)

این سیستم از ترکیب دو روش برای تولید توصیه‌های بهتر استفاده می‌کند:

1. **Collaborative Filtering** (60% وزن)
   - بر اساس رفتار کاربران مشابه
   - "کاربرانی که محصول X را خریده‌اند، محصول Y را هم خریده‌اند"

2. **Content-Based Filtering** (40% وزن)
   - بر اساس ویژگی‌های محصولات
   - "محصولات مشابه محصولاتی که شما خریده‌اید"

### مزایا

- ✅ **دقت بالا**: ترکیب دو روش برای نتایج بهتر
- ✅ **سرعت**: استفاده از Redis برای caching (< 1ms)
- ✅ **Scalable**: پشتیبانی از 200K+ کاربر
- ✅ **Flexible**: REST API برای استفاده در هر پلتفرم

---

## 🚀 نحوه استفاده

### مرحله 1: تست اتصال (اختیاری)

```bash
cd /path/to/recommendation
source venv/bin/activate
python test_db_connection.py
```

### مرحله 2: تولید توصیه‌ها

#### گزینه A: تست با 1000 کاربر (توصیه می‌شود) ⭐

```bash
python generate_recommendations.py --sample 1000
```

**زمان تخمینی:** 2-5 دقیقه

**مناسب برای:**
- تست اولیه سیستم
- بررسی عملکرد
- صرفه‌جویی در زمان

#### گزینه B: تولید برای همه کاربران

```bash
python generate_recommendations.py --all
# یا بدون آرگومنت
python generate_recommendations.py
```

**زمان تخمینی:** 15-45 دقیقه

**این اسکریپت:**
1. کاربران، محصولات و سفارشات را بارگذاری می‌کند
2. مدل‌های Collaborative و Content-Based را آموزش می‌دهد
3. برای همه کاربران توصیه تولید می‌کند (20 توصیه به ازای هر کاربر)
4. نتایج را در فایل‌های زیر ذخیره می‌کند:
   - `storage/app/recommendation/user_recommendations_YYYYMMDD_HHMMSS.parquet`
   - `storage/app/recommendation/user_recommendations_YYYYMMDD_HHMMSS.csv`
5. توصیه‌ها را در Redis ذخیره می‌کند

#### سایر گزینه‌ها:

```bash
# تست سریع (30 ثانیه)
python generate_recommendations.py --sample 100

# تست متوسط
python generate_recommendations.py --sample 5000

# مشاهده راهنما
python generate_recommendations.py --help
```

### مرحله 3: بررسی نتایج

```bash
# نمایش فایل‌های ایجاد شده
ls -lh storage/app/recommendation/user_recommendations_*

# مشاهده چند خط اول CSV
head -20 storage/app/recommendation/user_recommendations_*.csv

# یا با Python
python -c "
import polars as pl
df = pl.read_csv('storage/app/recommendation/user_recommendations_*.csv')
print(f'تعداد کل توصیه‌ها: {len(df)}')
print(f'تعداد کاربران: {df[\"user_id\"].n_unique()}')
print(df.head(20))
"
```

---

## 🌐 راه‌اندازی API

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
# روش 1: استفاده از run_recommendation.py (توصیه می‌شود)
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

**دسترسی به مستندات Swagger:**
- 📖 http://localhost:8000/docs
- 🔍 http://localhost:8000/health

---

## 📡 API Endpoints

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

**⚠️ توجه:** این endpoint می‌تواند 10-45 دقیقه طول بکشد!

---

## 🔗 استفاده در زبان‌های مختلف

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

## 🔧 ساختار پروژه

### فایل‌های اصلی:

```
recommendation/
├── generate_recommendations.py    ⭐ اسکریپت اصلی (با قابلیت --sample)
├── test_db_connection.py          ✅ تست اتصال
├── recommendation_api.py          ✅ FastAPI سرور
├── run_recommendation.py          ✅ CLI tool
├── hybrid_recommender.py          ✅ سیستم ترکیبی
├── collaborative_filtering.py     ✅ الگوریتم CF
├── content_based_filtering.py      ✅ الگوریتم CBF
├── recommendation_storage.py      ✅ مدیریت Redis
├── object_loader.py               ✅ بارگذاری object-based
├── dataframe_loader.py            ✅ بارگذاری dataframe-based
├── models.py                      ✅ مدل‌های داده
├── settings.py                    ✅ تنظیمات
├── pipeline.py                    ✅ پایپلاین Matomo
├── matomo_client.py              ✅ کلاینت Matomo
├── examples_usage.py              ✅ مثال‌های استفاده
└── README.md                      ✅ راهنمای سریع
```

### فایل‌های حذف شده:
- `test_connection.py` (قدیمی - جایگزین: `test_db_connection.py`)
- `run_generate.sh` (غیرضروری)

---

## 📁 ساختار فایل خروجی

فایل CSV شامل ستون‌های زیر است:

| ستون | توضیح |
|------|-------|
| `user_id` | شناسه کاربر |
| `product_id` | شناسه محصول توصیه شده |
| `score` | امتیاز توصیه (هرچه بالاتر، بهتر) |
| `rank` | رتبه توصیه (1 = بهترین) |
| `confidence` | میزان اطمینان (0-1) |
| `reason` | دلیل توصیه |
| `collaborative_details` | جزئیات Collaborative (JSON) |
| `generated_at` | زمان تولید توصیه |

---

## 💻 نیازمندی‌های منابع

با توجه به آمار فعلی سیستم:
- **224,959 کاربر**
- **36,114 محصول فعال**
- **4,499,180 توصیه (20 به ازای هر کاربر)**

### 📊 حجم حافظه مورد نیاز

```
حجم فعلی در Redis:  ~1.6 GB
با Overhead:         ~2.4 GB
RAM مورد نیاز:       4 GB (توصیه می‌شود)
```

### 💰 توصیه سرور برای شروع

**گزینه 1: کوچک (توصیه می‌شود):**
- RAM: 4 GB
- CPU: 2 vCPU
- Storage: 20 GB
- **هزینه:** ~$18-24/month

**گزینه 2: متوسط (رشد 1-3 سال):**
- RAM: 8 GB
- CPU: 4 vCPU  
- Storage: 50 GB
- **هزینه:** ~$36-48/month

### 📈 پیش‌بینی رشد

| دوره | کاربران | حجم Redis | RAM مورد نیاز |
|------|---------|-----------|---------------|
| فعلی | 224K | 1.6 GB | 4 GB |
| 1 سال | 337K | 2.4 GB | 4-8 GB |
| 2 سال | 450K | 3.2 GB | 8 GB |
| 3 سال | 675K | 4.8 GB | 8-12 GB |

> 📄 برای جزئیات بیشتر، فایل `RESOURCE_REQUIREMENTS.md` را مطالعه کنید.

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

---

## 🔄 بازآموزی دوره‌ای

توصیه می‌شود این اسکریپت را به صورت دوره‌ای (مثلاً هفتگی) اجرا کنید:

```bash
# اضافه کردن به crontab برای اجرای هفتگی (هر شنبه ساعت 2 صبح)
0 2 * * 6 cd /path/to/recommendation && source venv/bin/activate && python generate_recommendations.py >> logs/recommendations.log 2>&1
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

### مشکل: "هیچ توصیه‌ای تولید نشد"

احتمالاً:
- تعداد سفارشات کافی نیست (حداقل 100 سفارش نیاز است)
- بازه زمانی خیلی کوتاه است
- کاربران سفارش ثبت نکرده‌اند

**راه حل:** اسکریپت به طور خودکار بازه زمانی را افزایش می‌دهد

### مشکل: فرآیند خیلی کند است

دلایل احتمالی:
- تعداد زیاد کاربران (224K کاربر)
- تعداد زیاد محصولات (36K محصول)
- محاسبات ماتریس شباهت

**راه حل:**
- صبر کنید (10-30 دقیقه)
- یا می‌توانید تعداد کاربران را با `--sample` محدود کنید

---

## 📧 استفاده در Laravel

### روش 1: استفاده از Redis (توصیه می‌شود - سریع‌ترین) ⚡

```php
<?php
use Illuminate\Support\Facades\Redis;

// دریافت توصیه‌های کاربر از Redis
$userId = 123;
$key = "recommendation:{$userId}";
$recommendations = json_decode(Redis::get($key), true);

if ($recommendations) {
    // نمایش توصیه‌ها
    foreach ($recommendations as $rec) {
        echo "Product ID: {$rec['product_id']}\n";
        echo "Score: {$rec['score']}\n";
        echo "Reason: {$rec['reason']}\n";
        echo "Confidence: {$rec['confidence']}\n";
        
        // نمایش جزئیات Collaborative اگر موجود باشد
        if (!empty($rec['collaborative_details'])) {
            $details = json_decode($rec['collaborative_details'], true);
            echo "Similar Users: " . $details['total_similar_users'] . "\n";
            
            foreach ($details['similar_users'] as $user) {
                echo "  - User {$user['user_id']}: {$user['similarity_percent']}% similar\n";
            }
        }
    }
} else {
    // Fallback: دریافت از CSV یا تولید مستقیم
    echo "توصیه‌ای در Redis موجود نیست";
}
```

**مزایا Redis:**
- ⚡ سرعت بالا (O(1) read/write)
- 💾 حافظه بهینه
- 🔄 TTL خودکار (7 روز)
- ✅ بهترین گزینه برای caching

### روش 2: استفاده از فایل CSV (fallback)

```php
<?php
use Illuminate\Support\Facades\DB;

// خواندن توصیه‌ها برای یک کاربر
$userId = 123;
$csv = storage_path('app/recommendation/user_recommendations_latest.csv');

$recommendations = collect(array_map('str_getcsv', file($csv)))
    ->slice(1) // حذف header
    ->map(function ($row) {
        return [
            'user_id' => $row[0],
            'product_id' => $row[1],
            'score' => $row[2],
            'rank' => $row[3],
            'confidence' => $row[4],
            'reason' => $row[5],
            'collaborative_details' => $row[6] ?? null,
        ];
    })
    ->where('user_id', $userId)
    ->take(10);
```

### روش 3: استفاده از API (REST)

```php
<?php
use Illuminate\Support\Facades\Http;

$response = Http::get('http://localhost:8000/recommendations/123', [
    'limit' => 10
]);

$recommendations = $response->json();
```

### تنظیمات Redis در `.env` Laravel

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=null
REDIS_TTL_SECONDS=604800  # 7 days
```

---

## 🗄️ استفاده از Redis

سیستم به صورت خودکار توصیه‌ها را در Redis ذخیره می‌کند. Redis انتخاب بهتری نسبت به MongoDB است:

| ویژگی | Redis ✅ | MongoDB |
|-------|---------|---------|
| سرعت | خیلی سریع (O(1)) | سریع |
| حافظه | بهینه | متوسط |
| TTL | ✅ خودکار | ❌ دستی |
| پیچیدگی | ساده | پیچیده‌تر |
| مناسب برای | Caching | Analytics |

**ساختار کلیدهای Redis:**
- `recommendation:{user_id}` → JSON array با 20 توصیه
- `recommendation_meta:{user_id}` → metadata (تاریخ، تعداد، etc.)

**نحوه نصب Redis:**
```bash
# macOS
brew install redis
brew services start redis

# Linux (Ubuntu/Debian)
sudo apt install redis-server
sudo systemctl start redis

# تست اتصال
redis-cli ping  # باید PONG برگرداند
```

### نحوه استفاده از توصیه‌های ذخیره شده

#### در Python:

```python
from recommendation_storage import get_storage

# دریافت توصیه‌ها برای یک کاربر
storage = get_storage()
recommendations = storage.get_recommendations(user_id=123)

if recommendations:
    for rec in recommendations:
        print(f"Product: {rec['product_id']}")
        print(f"Score: {rec['score']}")
        print(f"Reason: {rec['reason']}")
        
        # نمایش جزئیات Collaborative
        if rec.get('collaborative_details'):
            import json
            details = json.loads(rec['collaborative_details'])
            print(f"Similar Users: {details['total_similar_users']}")
```

#### بررسی سریع (بدون دریافت کامل):

```python
# بررسی وجود توصیه‌ها (سریع)
exists = storage.exists(user_id=123)

# دریافت metadata
metadata = storage.get_metadata(user_id=123)
print(f"تولید شده: {metadata['generated_at']}")
```

> 💡 برای مثال‌های بیشتر، فایل `examples_usage.py` را مشاهده کنید.

---

## 💡 نکات مهم

1. **توصیه شدید:** برای اولین بار با `--sample 1000` شروع کنید
2. دیتابیس شما **224,959 کاربر** دارد - پردازش کامل زمان‌بر است
3. سیستم فقط **محصولات فعال با موجودی** را توصیه می‌کند
4. سیستم محصولاتی که کاربر قبلاً خریده را توصیه نمی‌کند
5. فایل CSV را می‌توانید به راحتی در Laravel بخوانید و استفاده کنید
6. زمان تخمینی: 100 کاربر ≈ 30 ثانیه، 1000 کاربر ≈ 3-5 دقیقه، همه کاربران ≈ 15-45 دقیقه
7. **Postman Collection:** برای تست API، فایل `Recommendation_API.postman_collection.json` را import کنید

---

## 🎯 مراحل بعدی

1. ✅ اجرای تست: `python generate_recommendations.py --sample 1000`
2. ✅ بررسی فایل CSV خروجی
3. ✅ اگر نتیجه مناسب بود، اجرای کامل: `python generate_recommendations.py --all`
4. ✅ استفاده از توصیه‌ها در Laravel
5. ✅ نمایش توصیه‌ها به کاربران
6. ✅ تنظیم cron job برای بازآموزی دوره‌ای

---

**موفق باشید! 🚀**

