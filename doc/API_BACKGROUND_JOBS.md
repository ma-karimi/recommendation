# API برای تولید توصیه در پس‌زمینه

این مستندات نحوه استفاده از API برای تولید توصیه برای کاربران جدید در پس‌زمینه را توضیح می‌دهد.

## 🔄 فرآیند کار

1. **ارسال درخواست**: درخواست تولید توصیه را ارسال می‌کنید
2. **دریافت فوری**: API فوراً یک `job_id` برمی‌گرداند
3. **اجرای پس‌زمینه**: فرآیند تولید توصیه در پس‌زمینه شروع می‌شود
4. **بررسی وضعیت**: می‌توانید وضعیت job را بررسی کنید

## 📡 Endpoints

### 1. تولید توصیه (POST)

**Endpoint:** `POST /generate-recommendations`

**Request Body:**
```json
{
  "user_ids": [12345, 67890, 11111],
  "top_k": 20
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "درخواست ثبت شد و در حال پردازش است",
  "created_at": "2025-12-01T10:30:00",
  "total_users": 3,
  "processed_users": 0,
  "failed_users": 0
}
```

**مثال با curl:**
```bash
curl -X POST "http://localhost:8000/generate-recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_ids": [12345, 67890],
    "top_k": 20
  }'
```

**مثال با Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/generate-recommendations",
    json={
        "user_ids": [12345, 67890, 11111],
        "top_k": 20
    }
)

job = response.json()
job_id = job["job_id"]
print(f"Job created: {job_id}")
```

### 2. بررسی وضعیت Job (GET)

**Endpoint:** `GET /job-status/{job_id}`

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "message": "در حال پردازش...",
  "created_at": "2025-12-01T10:30:00",
  "started_at": "2025-12-01T10:30:01",
  "completed_at": null,
  "total_users": 3,
  "processed_users": 1,
  "failed_users": 0,
  "error": null
}
```

**وضعیت‌های ممکن:**
- `pending`: در انتظار شروع
- `running`: در حال اجرا
- `completed`: با موفقیت تمام شد
- `failed`: خطا رخ داد

**مثال با curl:**
```bash
curl "http://localhost:8000/job-status/550e8400-e29b-41d4-a716-446655440000"
```

**مثال با Python:**
```python
import requests
import time

job_id = "550e8400-e29b-41d4-a716-446655440000"

while True:
    response = requests.get(f"http://localhost:8000/job-status/{job_id}")
    job = response.json()
    
    print(f"Status: {job['status']}, Processed: {job['processed_users']}/{job['total_users']}")
    
    if job['status'] in ['completed', 'failed']:
        break
    
    time.sleep(2)  # 2 ثانیه صبر کن
```

### 3. لیست Job ها (GET)

**Endpoint:** `GET /jobs?limit=10`

**Response:**
```json
[
  {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "message": "توصیه‌ها با موفقیت برای 3 کاربر تولید شد",
    "created_at": "2025-12-01T10:30:00",
    "started_at": "2025-12-01T10:30:01",
    "completed_at": "2025-12-01T10:32:15",
    "total_users": 3,
    "processed_users": 3,
    "failed_users": 0,
    "error": null
  }
]
```

**مثال:**
```bash
curl "http://localhost:8000/jobs?limit=5"
```

### 4. حذف Job (DELETE)

**Endpoint:** `DELETE /job/{job_id}`

**نکته:** فقط job های `completed` یا `failed` را می‌توان حذف کرد.

**مثال:**
```bash
curl -X DELETE "http://localhost:8000/job/550e8400-e29b-41d4-a716-446655440000"
```

## 📝 مثال کامل

```python
import requests
import time

# 1. ایجاد job
response = requests.post(
    "http://localhost:8000/generate-recommendations",
    json={
        "user_ids": [12345, 67890, 11111],
        "top_k": 20
    }
)
job = response.json()
job_id = job["job_id"]
print(f"✅ Job created: {job_id}")

# 2. بررسی وضعیت
while True:
    response = requests.get(f"http://localhost:8000/job-status/{job_id}")
    job = response.json()
    
    status = job['status']
    print(f"📊 Status: {status}")
    
    if status == 'completed':
        print(f"✅ Job completed! Processed {job['processed_users']} users")
        break
    elif status == 'failed':
        print(f"❌ Job failed: {job.get('error', 'Unknown error')}")
        break
    else:
        print(f"⏳ Processing... {job['processed_users']}/{job['total_users']} users")
        time.sleep(2)

# 3. بررسی توصیه‌ها (بعد از اتمام)
for user_id in [12345, 67890, 11111]:
    response = requests.get(f"http://localhost:8000/recommendations/{user_id}")
    recommendations = response.json()
    print(f"User {user_id}: {len(recommendations)} recommendations")
```

## ⚠️ نکات مهم

1. **Timeout**: درخواست API فوراً برمی‌گردد و timeout نمی‌دهد
2. **Background Processing**: فرآیند در پس‌زمینه اجرا می‌شود
3. **Job Tracking**: وضعیت job ها در memory نگهداری می‌شود (بعد از restart از بین می‌رود)
4. **مدل باید train شده باشد**: قبل از استفاده، مدل باید حداقل یک بار train شده باشد

## 🔍 بررسی وضعیت سیستم

```bash
# بررسی سلامت سیستم
curl "http://localhost:8000/health"

# آمار سیستم
curl "http://localhost:8000/stats"
```

## 🚀 استفاده در Production

برای استفاده در production، توصیه می‌شود:

1. **Job Storage**: از Redis یا دیتابیس برای ذخیره وضعیت job ها استفاده کنید
2. **Queue System**: از Celery یا RQ برای مدیریت job ها استفاده کنید
3. **Monitoring**: از Prometheus یا Grafana برای مانیتورینگ استفاده کنید

## 📚 مستندات کامل API

برای مشاهده مستندات کامل و تست API، به آدرس زیر بروید:

```
http://localhost:8000/docs
```

