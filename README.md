# سیستم توصیه محصولات (Product Recommendation System)

<div dir="rtl">

سیستم توصیه محصولات هوشمند مبتنی بر **Hybrid Recommender** که از ترکیب **Collaborative Filtering** و **Content-Based Filtering** برای تولید توصیه‌های شخصی‌سازی شده استفاده می‌کند.

</div>

---

## ✨ ویژگی‌ها

- 🎯 **Hybrid Recommender**: ترکیب Collaborative و Content-Based Filtering
- ⚡ **Redis Cache**: سرعت بالا با ذخیره‌سازی در Redis (< 1ms)
- 🚀 **REST API**: FastAPI با مستندات Swagger کامل
- 📊 **Scalable**: پشتیبانی از 200K+ کاربر
- 🔄 **Auto Retrain**: امکان بازآموزی مدل
- 📈 **Insights**: بینش‌های کاربر و تحلیل رفتار

---

## 🚀 شروع سریع

### پیش‌نیازها

- Python 3.9+
- Redis 6.0+
- MySQL/MariaDB Database
- 4 GB RAM (حداقل)

### نصب

```bash
# کلون کردن پروژه
git clone <repository-url>
cd recommendation

# ایجاد virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# نصب dependencies
pip install -r requirements.txt
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# نصب Redis
# macOS
brew install redis
brew services start redis

# Linux
sudo apt install redis-server
sudo systemctl start redis
```

### تنظیمات

فایل `.env` را ایجاد و تنظیم کنید:

```env
# Database
RECO_DB_URL=mysql+pymysql://user:password@host:port/database

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_TTL_SECONDS=604800

# Matomo (اختیاری)
MATOMO_BASE_URL=https://analytics.example.com
MATOMO_SITE_ID=1
MATOMO_TOKEN_AUTH=your_token
MATOMO_VERIFY_SSL=true
```

### استفاده

```bash
# 1. تست اتصال (اختیاری)
python test_db_connection.py

# 2. تولید توصیه‌ها (تست با 1000 کاربر)
python generate_recommendations.py --sample 1000

# 3. راه‌اندازی API
python run_recommendation.py api --host 0.0.0.0 --port 8000

# 4. مشاهده مستندات API
open http://localhost:8000/docs
```

---

## 📚 مستندات

| سند | توضیح |
|-----|-------|
| [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) | **🎓 راهنمای یادگیری** - آموزش کامل برای مبتدیان: مفاهیم، الگوریتم‌ها، فرآیندها و مثال‌ها |
| [docs/GUIDE.md](docs/GUIDE.md) | **راهنمای کامل** - تمام جزئیات استفاده، API، تنظیمات، مثال‌ها و troubleshooting |
| [docs/RESOURCE_REQUIREMENTS.md](docs/RESOURCE_REQUIREMENTS.md) | نیازمندی‌های منابع و پیش‌بینی رشد |
| [docs/Recommendation_API.postman_collection.json](docs/Recommendation_API.postman_collection.json) | مجموعه Postman برای تست API |

---

## 🏗️ ساختار پروژه

```
recommendation/
├── recommendation_api.py       # FastAPI REST API
├── hybrid_recommender.py       # Hybrid Recommender
├── collaborative_filtering.py  # Collaborative Filtering
├── content_based_filtering.py # Content-Based Filtering
├── generate_recommendations.py # تولید توصیه‌ها
├── recommendation_storage.py  # مدیریت Redis
├── object_loader.py           # بارگذاری داده‌ها (object-based)
├── dataframe_loader.py       # بارگذاری داده‌ها (dataframe-based)
├── models.py                  # مدل‌های داده
├── settings.py                # تنظیمات
├── run_recommendation.py      # CLI tool
├── test_db_connection.py     # تست اتصال
└── examples_usage.py         # مثال‌های استفاده
```

---

## 📡 API Endpoints

### اصلی

- `GET /` - صفحه اصلی
- `GET /health` - بررسی سلامت سیستم
- `GET /stats` - آمار سیستم

### توصیه‌ها

- `GET /recommendations/{user_id}` - دریافت توصیه‌ها ⭐
- `GET /insights/{user_id}` - بینش‌های کاربر
- `GET /popular` - محصولات محبوب
- `GET /similar/{product_id}` - محصولات مشابه

### مدیریت

- `POST /retrain` - بازآموزی مدل

**📖 مستندات کامل API:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 💡 مثال استفاده

### Python

```python
import requests

response = requests.get("http://localhost:8000/recommendations/123?limit=10")
recommendations = response.json()

for rec in recommendations:
    print(f"Product: {rec['product_id']}")
    print(f"Score: {rec['score']}")
    print(f"Reason: {rec['reason']}")
```

### Laravel / PHP

```php
use Illuminate\Support\Facades\Http;

$response = Http::get('http://localhost:8000/recommendations/123', [
    'limit' => 10
]);

$recommendations = $response->json();
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/recommendations/123?limit=10');
const recommendations = await response.json();
```

---

## 🔧 دستورات CLI

```bash
# آموزش مدل
python run_recommendation.py train

# اجرای API
python run_recommendation.py api --host 0.0.0.0 --port 8000

# دریافت توصیه برای یک کاربر
python run_recommendation.py recommend <user_id> --limit 10

# تولید توصیه‌ها
python generate_recommendations.py --sample 1000  # تست
python generate_recommendations.py --all          # همه کاربران
```

---

## 📊 آمار فعلی

```
✅ کاربران: 224,959
✅ محصولات: 36,114 (فعال)
✅ سفارشات: 80,737
✅ توصیه‌ها: ~4.5M (20 به ازای هر کاربر)
```

---

## ⚙️ تکنولوژی‌ها

- **Python 3.9+**
- **FastAPI** - REST API Framework
- **Redis** - Cache Layer
- **NumPy / SciPy** - محاسبات عددی
- **scikit-learn** - Machine Learning
- **Polars** - پردازش داده‌ها
- **SQLAlchemy** - ORM

---

## 🔒 امنیت

برای Production:

1. اضافه کردن Authentication (API Key / JWT)
2. استفاده از HTTPS
3. Rate Limiting
4. Input Validation
5. Logging و Monitoring

---

## 📈 Performance

- **Redis Read:** < 1ms ⚡
- **Fallback Mode:** ~50-100ms
- **API Response Time:** < 5ms (با Redis)
- **Throughput:** 1000+ requests/second

---

## 🐛 Troubleshooting

### مشکل اتصال به Redis

```bash
# بررسی Redis
redis-cli ping  # باید PONG برگرداند

# راه‌اندازی Redis
brew services start redis  # macOS
sudo systemctl start redis # Linux
```

### مشکل اتصال به Database

```bash
# تست اتصال
python test_db_connection.py

# بررسی تنظیمات
cat .env | grep RECO_DB_URL
```

### مشکل کند بودن

- بررسی اینکه Redis در حال اجرا است
- استفاده از `--sample` برای تست اولیه
- بررسی logs برای خطاها

---

## 📞 پشتیبانی

- **مستندات کامل:** [docs/GUIDE.md](docs/GUIDE.md)
- **Issues:** برای گزارش مشکلات
- **Examples:** `examples_usage.py`

---

## 📄 License

[مشخص کنید]

---

## 🎯 مراحل بعدی

1. ✅ راه‌اندازی Redis و Database
2. ✅ تولید توصیه‌های اولیه (`--sample 1000`)
3. ✅ بررسی نتایج
4. ✅ راه‌اندازی API
5. ✅ ادغام با Laravel/Backend
6. ✅ تنظیم Cron Job برای بازآموزی دوره‌ای

---

**موفق باشید! 🚀**

برای جزئیات کامل، [docs/GUIDE.md](docs/GUIDE.md) را مطالعه کنید.
