# راهنمای تولید توصیه برای کاربران مشخص

این راهنما نحوه تولید توصیه برای یک یا چند کاربر مشخص را توضیح می‌دهد.

## پیش‌نیازها

قبل از استفاده از این قابلیت، باید مدل را حداقل یک بار train کنید:

```bash
python generate_recommendations.py --sample 100
```

یا برای train کردن کامل:

```bash
python generate_recommendations.py --all
```

## روش‌های استفاده

### 1. تولید توصیه برای یک کاربر

```bash
python generate_recommendations.py --user 12345
```

### 2. تولید توصیه برای چند کاربر (از command line)

```bash
python generate_recommendations.py --users 123 456 789 1011
```

### 3. تولید توصیه برای چند کاربر (از فایل)

ابتدا یک فایل متنی با لیست user_id ها ایجاد کنید:

```bash
# فایل user_ids.txt
12345
67890
11111
22222
```

سپس:

```bash
python generate_recommendations.py --users-file user_ids.txt
```

**نکته:** فایل می‌تواند شامل کامنت باشد (خطوطی که با `#` شروع می‌شوند نادیده گرفته می‌شوند).

### 4. تنظیم تعداد توصیه برای هر کاربر

```bash
python generate_recommendations.py --user 12345 --top-k 50
```

## مثال‌های کامل

### مثال 1: یک کاربر جدید

```bash
# تولید 20 توصیه برای کاربر 12345
python generate_recommendations.py --user 12345
```

### مثال 2: چند کاربر جدید

```bash
# تولید توصیه برای 3 کاربر
python generate_recommendations.py --users 100 200 300 --top-k 30
```

### مثال 3: استفاده از فایل

```bash
# ایجاد فایل
echo -e "12345\n67890\n11111" > new_users.txt

# تولید توصیه
python generate_recommendations.py --users-file new_users.txt
```

## خروجی

توصیه‌ها در دو فرمت ذخیره می‌شوند:

1. **Parquet**: `storage/app/recommendation/user_recommendations_YYYYMMDD_HHMMSS.parquet`
2. **CSV**: `storage/app/recommendation/user_recommendations_YYYYMMDD_HHMMSS.csv`
3. **Redis**: (اگر Redis در دسترس باشد)

## نکات مهم

1. **مدل باید train شده باشد**: قبل از استفاده، حتماً مدل را train کنید.
2. **استفاده از storage**: این قابلیت از مدل‌های train شده در DuckDB استفاده می‌کند و نیازی به train مجدد ندارد.
3. **سرعت**: تولید توصیه برای کاربران مشخص بسیار سریع‌تر از train کردن کامل است.
4. **حافظه**: این روش حافظه کمتری مصرف می‌کند چون فقط برای کاربران مشخص کار می‌کند.

## خطاهای رایج

### خطا: "ANN index یافت نشد"

**راه حل:** ابتدا مدل را train کنید:
```bash
python generate_recommendations.py --sample 100
```

### خطا: "Could not load user profiles"

این خطا معمولاً مشکلی ایجاد نمی‌کند و فقط یک warning است.

### خطا: "No recommendations generated"

این می‌تواند به دلایل زیر باشد:
- کاربر هیچ تعاملی نداشته است
- محصولات موجود نیستند
- مدل به درستی train نشده است

## مقایسه با روش‌های دیگر

| روش | سرعت | حافظه | کاربرد |
|-----|------|-------|--------|
| `--all` | کند | زیاد | Train کامل |
| `--sample N` | متوسط | متوسط | تست و train |
| `--user ID` | سریع | کم | یک کاربر |
| `--users ID1 ID2 ...` | سریع | کم | چند کاربر |
| `--users-file FILE` | سریع | کم | لیست کاربران |

