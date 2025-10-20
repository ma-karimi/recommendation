# سیستم توصیه محصولات (Product Recommendation System)

سیستم توصیه محصولات کاربرمحور که از ترکیب الگوریتم‌های Collaborative Filtering و Content-Based Filtering استفاده می‌کند.

## ویژگی‌ها

- **Collaborative Filtering**: توصیه بر اساس رفتار کاربران مشابه
- **Content-Based Filtering**: توصیه بر اساس ویژگی‌های محصولات
- **سیستم ترکیبی**: ترکیب هوشمند نتایج دو روش
- **API RESTful**: رابط برنامه‌نویسی برای دریافت توصیه‌ها
- **پشتیبانی از Matomo**: تحلیل رفتار کاربران از داده‌های Matomo
- **فیلترهای تجاری**: حذف محصولات نامناسب (ناموجود، غیرفعال)

## نصب و راه‌اندازی

### 1. نصب وابستگی‌ها
```bash
cd /Users/mohammad/Projects/srico/rochi/recommendation
source venv/bin/activate
pip install -r requirements.txt
```

### 2. تنظیم متغیرهای محیطی
فایل `.env` را در ریشه پروژه ایجاد کنید:

```env
# Matomo Analytics
MATOMO_BASE_URL=https://your-matomo.example.com
MATOMO_SITE_ID=1
MATOMO_TOKEN_AUTH=your_token_here
MATOMO_VERIFY_SSL=true

# Database
RECO_DB_URL=mysql+pymysql://user:pass@host:3306/dbname?charset=utf8mb4
# یا استفاده از متغیرهای Laravel:
# DB_CONNECTION=mysql
# DB_HOST=127.0.0.1
# DB_PORT=3306
# DB_DATABASE=dbname
# DB_USERNAME=user
# DB_PASSWORD=pass

# Storage Path
STORAGE_PATH=/path/to/your/storage
```

### 3. اجرای سیستم

#### آموزش مدل
```bash
python run_recommendation.py train
```

#### اجرای API سرور
```bash
python run_recommendation.py api --host 0.0.0.0 --port 8000
```

#### دریافت توصیه‌های کاربر
```bash
python run_recommendation.py recommend 123 --limit 10
```

## استفاده از API

### دریافت توصیه‌های کاربر
```bash
curl "http://localhost:8000/recommendations/123?limit=10"
```

### دریافت بینش‌های کاربر
```bash
curl "http://localhost:8000/insights/123"
```

### دریافت محصولات محبوب
```bash
curl "http://localhost:8000/popular?limit=10"
```

### دریافت محصولات مشابه
```bash
curl "http://localhost:8000/similar/456?limit=5"
```

### بازآموزی مدل
```bash
curl -X POST "http://localhost:8000/retrain"
```

## ساختار پروژه

```
recommendation/
├── models.py                    # مدل‌های داده
├── data_loader.py              # بارگذاری داده‌ها از DB
├── collaborative_filtering.py  # الگوریتم Collaborative Filtering
├── content_based_filtering.py  # الگوریتم Content-Based Filtering
├── hybrid_recommender.py       # سیستم ترکیبی
├── recommendation_api.py       # API RESTful
├── run_recommendation.py       # اسکریپت اجرا
├── pipeline.py                 # پایپلاین Matomo
├── settings.py                 # تنظیمات
└── requirements.txt            # وابستگی‌ها
```

## الگوریتم‌های استفاده شده

### Collaborative Filtering
- محاسبه شباهت کاربران با کسینوس
- پیش‌بینی امتیاز محصولات
- پیدا کردن کاربران مشابه

### Content-Based Filtering
- استخراج ویژگی‌های محصولات (TF-IDF)
- محاسبه شباهت محصولات
- ساخت پروفایل کاربران

### سیستم ترکیبی
- ترکیب نتایج دو روش
- اعمال فیلترهای تجاری
- رتبه‌بندی نهایی

## مثال استفاده

```python
from hybrid_recommender import HybridRecommender

# آموزش مدل
recommender = HybridRecommender()
recommender.train()

# دریافت توصیه‌ها
recommendations = recommender.get_recommendations(user_id=123, top_k=10)

for rec in recommendations:
    print(f"محصول {rec.product_id}: {rec.score:.2f} - {rec.reason}")
```

## مستندات API

پس از اجرای سرور، مستندات کامل API در آدرس زیر در دسترس است:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## نکات مهم

1. **داده‌های کافی**: سیستم برای عملکرد بهتر نیاز به داده‌های کافی دارد
2. **بازآموزی دوره‌ای**: مدل را به صورت دوره‌ای بازآموزی کنید
3. **مانیتورینگ**: عملکرد سیستم را نظارت کنید
4. **بهینه‌سازی**: برای سیستم‌های بزرگ، از تکنیک‌های بهینه‌سازی استفاده کنید

## عیب‌یابی

### خطای اتصال به پایگاه داده
- بررسی متغیرهای محیطی `RECO_DB_URL` یا `DB_*`
- اطمینان از دسترسی به پایگاه داده

### خطای Matomo
- بررسی `MATOMO_BASE_URL` و `MATOMO_TOKEN_AUTH`
- اطمینان از دسترسی به API Matomo

### خطای حافظه
- کاهش تعداد محصولات یا کاربران
- استفاده از نمونه‌گیری (sampling)

## پشتیبانی

برای سوالات و مشکلات، لطفاً با تیم توسعه تماس بگیرید.
