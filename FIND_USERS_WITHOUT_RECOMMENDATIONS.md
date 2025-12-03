# پیدا کردن کاربران بدون توصیه

این راهنما نحوه پیدا کردن کاربرانی که هنوز توصیه برایشان ایجاد نشده است را توضیح می‌دهد.

## 🎯 روش‌های استفاده

### 1. از طریق Command Line

#### پیدا کردن همه کاربران بدون توصیه:
```bash
python generate_recommendations.py --find-without-recommendations
```

#### محدود کردن تعداد کاربران برای بررسی:
```bash
python generate_recommendations.py --find-without-recommendations --sample 1000
```

#### ذخیره لیست در فایل:
```bash
python generate_recommendations.py --find-without-recommendations --output-file users_without_recs.csv
```

**خروجی:**
- فایل CSV: `users_without_recs.csv`
- فایل TXT: `users_without_recs.txt` (فقط لیست user_id ها)

### 2. از طریق API

#### دریافت لیست کامل:
```bash
curl "http://localhost:8000/users-without-recommendations"
```

**Response:**
```json
{
  "total_users": 262090,
  "users_with_recommendations": 21162,
  "users_without_recommendations": 240928,
  "user_ids_without_recommendations": [123, 456, 789, ...],
  "percentage_with_recommendations": 8.07,
  "percentage_without_recommendations": 91.93
}
```

#### دریافت تعداد سریع (بدون لیست):
```bash
curl "http://localhost:8000/users-without-recommendations/count"
```

**Response:**
```json
{
  "total_users": 262090,
  "sample_size": 1000,
  "users_without_in_sample": 920,
  "estimated_percentage_without": 92.0,
  "estimated_users_without_recommendations": 241123,
  "note": "این یک تخمین است. برای لیست دقیق از /users-without-recommendations استفاده کنید"
}
```

#### محدود کردن تعداد:
```bash
curl "http://localhost:8000/users-without-recommendations?limit=1000"
```

### 3. از طریق Python

```python
from generate_recommendations import find_users_without_recommendations

# پیدا کردن همه کاربران بدون توصیه
users_without = find_users_without_recommendations()

print(f"تعداد کاربران بدون توصیه: {len(users_without)}")
print(f"10 کاربر اول: {users_without[:10]}")

# ذخیره در فایل
find_users_without_recommendations(
    output_file="users_without_recs.csv"
)

# محدود کردن بررسی
users_without = find_users_without_recommendations(limit=1000)
```

## 📊 مثال خروجی

```
================================================================================
جستجوی کاربران بدون توصیه
================================================================================

📥 بارگذاری کاربران از دیتابیس...
✅ 262090 کاربر بارگذاری شد

🔍 بررسی وجود توصیه‌ها در Redis...
✅ اتصال به Redis برقرار شد

📊 بررسی 262090 کاربر...
   بررسی شده: 1000/262090 (0.4%)
   بررسی شده: 2000/262090 (0.8%)
   ...
   بررسی شده: 262090/262090 (100.0%)

================================================================================
📊 نتایج:
================================================================================
   کل کاربران بررسی شده: 262,090
   کاربران با توصیه: 21,162 (8.1%)
   کاربران بدون توصیه: 240,928 (91.9%)
================================================================================

💾 لیست کاربران بدون توصیه در فایل ذخیره شد: users_without_recs.csv
💾 نسخه TXT نیز ذخیره شد: users_without_recs.txt

✅ 240928 کاربر بدون توصیه پیدا شد
```

## 🔄 استفاده بعدی

بعد از پیدا کردن کاربران بدون توصیه، می‌توانید برای آن‌ها توصیه تولید کنید:

### از Command Line:
```bash
# اگر فایل TXT دارید
python generate_recommendations.py --users-file users_without_recs.txt

# یا لیست مستقیم
python generate_recommendations.py --users 123 456 789
```

### از API:
```python
import requests

# دریافت لیست کاربران بدون توصیه
response = requests.get("http://localhost:8000/users-without-recommendations?limit=100")
data = response.json()
user_ids = data["user_ids_without_recommendations"]

# تولید توصیه برای آن‌ها
response = requests.post(
    "http://localhost:8000/generate-recommendations",
    json={
        "user_ids": user_ids[:100],  # 100 کاربر اول
        "top_k": 20
    }
)
job_id = response.json()["job_id"]
print(f"Job created: {job_id}")
```

## ⚡ بهینه‌سازی

### برای مجموعه‌های بزرگ:

1. **استفاده از نمونه‌گیری:**
   ```bash
   # فقط 1000 کاربر را بررسی کن
   python generate_recommendations.py --find-without-recommendations --sample 1000
   ```

2. **استفاده از API count endpoint:**
   ```bash
   # سریع‌تر - فقط تعداد
   curl "http://localhost:8000/users-without-recommendations/count"
   ```

3. **Batch processing:**
   ```python
   # بررسی به صورت batch
   all_users = load_users_from_db()
   batch_size = 10000
   
   for i in range(0, len(all_users), batch_size):
       batch = all_users[i:i+batch_size]
       users_without = find_users_without_recommendations(limit=batch_size)
       # پردازش batch
   ```

## 📝 نکات مهم

1. **نیاز به Redis**: این قابلیت نیاز به اتصال به Redis دارد
2. **سرعت**: بررسی همه کاربران ممکن است زمان‌بر باشد (بسته به تعداد)
3. **Memory**: برای مجموعه‌های بزرگ، از `limit` استفاده کنید
4. **فایل خروجی**: فایل‌های CSV و TXT در مسیر مشخص شده ذخیره می‌شوند

## 🔍 بررسی دستی

اگر می‌خواهید دستی بررسی کنید:

```python
from recommendation_storage import get_storage

storage = get_storage()
user_id = 12345

if storage.exists(user_id):
    print(f"کاربر {user_id} توصیه دارد")
else:
    print(f"کاربر {user_id} توصیه ندارد")
```

## 📚 مستندات مرتبط

- `USER_RECOMMENDATIONS_GUIDE.md`: تولید توصیه برای کاربران مشخص
- `API_BACKGROUND_JOBS.md`: استفاده از API برای تولید توصیه

