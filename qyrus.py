# a=[1,2,3,4]
# b=a
# b.append(4)
# print(a)
# print(b)
# c=a.copy()
# c.append(4)
# print(a)
# print(c)

# squares=[x:x**2 for x in range(5)  ]
# print(squares)





# def func(a, *args, **kwargs):
#     print(a)
#     print(args)
#     print(kwargs)

# func(1, 2, 3, name="AI", role="Engineer")

# def my_decorator(func):
#     def wrapper():
#         print("Before")
#         func()
#         print("After")
#     return wrapper


# @my_decorator
# def say_hi():
#     print("hi")
# say_hi()



print("=" * 60)
print("EXCEPTION HANDLING IN PYTHON")
print("=" * 60)

# 1. BASIC TRY-EXCEPT
print("\n1. BASIC TRY-EXCEPT:")
try:
    result = 10 / 0  # This will cause ZeroDivisionError
except ZeroDivisionError:
    print("❌ Error: Cannot divide by zero!")

# 2. CATCHING GENERIC EXCEPTIONS
print("\n2. CATCHING MULTIPLE EXCEPTIONS:")
try:
    num = int("abc")  # ValueError
except ValueError:
    print("❌ Error: Invalid number format")
except ZeroDivisionError:
    print("❌ Error: Division by zero")

# 3. CATCHING ANY EXCEPTION
print("\n3. CATCHING ANY EXCEPTION:")
try:
    result = 10 / 0
except Exception as e:
    print(f"❌ Error occurred: {e}")
    print(f"Error type: {type(e).__name__}")

# 4. MULTIPLE EXCEPTIONS IN ONE EXCEPT
print("\n4. HANDLING MULTIPLE EXCEPTION TYPES:")
try:
    data = [1, 2, 3]
    print(data[10])  # IndexError
except (IndexError, KeyError, ValueError) as e:
    print(f"❌ Error: {e}")

# 5. ELSE BLOCK - Runs if NO exception occurred
print("\n5. TRY-EXCEPT-ELSE:")
try:
    num = int("42")
    print(f"✓ Number: {num}")
except ValueError:
    print("❌ Invalid number")
else:
    print("✓ No error occurred! else block executed")

# 6. FINALLY BLOCK - Always runs (cleanup code)
print("\n6. TRY-EXCEPT-FINALLY:")
try:
    file = open("nonexistent.txt", "r")
except FileNotFoundError:
    print("❌ File not found")
finally:
    print("✓ Finally block always executes (cleanup code)")

# 7. RAISING CUSTOM EXCEPTIONS
print("\n7. RAISING CUSTOM EXCEPTIONS:")
def check_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative!")
    elif age < 18:
        raise ValueError("Must be 18 or older")
    print(f"✓ Age accepted: {age}")

try:
    check_age(-5)
except ValueError as e:
    print(f"❌ {e}")

try:
    check_age(25)
except ValueError as e:
    print(f"❌ {e}")

# 8. CUSTOM EXCEPTION CLASS
print("\n8. CUSTOM EXCEPTION CLASS:")
class InsufficientFundsError(Exception):
    pass

class BankAccount:
    def __init__(self, balance):
        self.balance = balance
    
    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(f"Need ${amount}, but only have ${self.balance}")
        self.balance -= amount
        print(f"✓ Withdrew ${amount}. Remaining: ${self.balance}")

account = BankAccount(100)
try:
    account.withdraw(50)
    account.withdraw(100)  # More than available
except InsufficientFundsError as e:
    print(f"❌ {e}")

# 9. REAL WORLD EXAMPLE - API CALL WITH ERROR HANDLING
print("\n9. REAL WORLD EXAMPLE - ROBUST FUNCTION:")
def safe_divide(a, b):
    try:
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Both inputs must be numbers")
        if b == 0:
            raise ZeroDivisionError("Divisor cannot be zero")
        return a / b
    except (TypeError, ZeroDivisionError) as e:
        print(f"❌ Error: {e}")
        return None

print(f"Result: {safe_divide(10, 2)}")
print(f"Result: {safe_divide(10, 0)}")
print(f"Result: {safe_divide('a', 2)}")

# 10. REAL WORLD EXAMPLE - FILE PROCESSING
print("\n10. FILE PROCESSING WITH ERROR HANDLING:")
def process_file(filename):
    try:
        with open(filename, 'r') as f:
            data = f.read()
            print(f"✓ File '{filename}' processed successfully")
            return data
    except FileNotFoundError:
        print(f"❌ File '{filename}' not found")
        return None
    except IOError as e:
        print(f"❌ Cannot read file: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

process_file("nonexistent.txt")

# 11. NESTED EXCEPTIONS
print("\n11. NESTED EXCEPTION HANDLING:")
try:
    try:
        result = 10 / 0
    except ZeroDivisionError:
        print("❌ Inner exception caught")
        raise ValueError("Converting to different error")  # Raise new exception
except ValueError as e:
    print(f"❌ Outer exception caught: {e}")

# 12. RETRYING ON EXCEPTION
print("\n12. RETRY MECHANISM:")
def unstable_operation(attempt):
    if attempt < 3:
        raise ConnectionError(f"Attempt {attempt} failed")
    return "Success!"

max_retries = 5
for attempt in range(1, max_retries + 1):
    try:
        result = unstable_operation(attempt)
        print(f"✓ {result} on attempt {attempt}")
        break
    except ConnectionError as e:
        print(f"❌ {e}. Retrying...")
        if attempt == max_retries:
            print("❌ All retries exhausted!")

print("\n" + "=" * 60)
print("KEY EXCEPTION TYPES:")
print("=" * 60)
print("- ValueError: Invalid value for data type")
print("- TypeError: Wrong data type")
print("- ZeroDivisionError: Division by zero")
print("- IndexError: List index out of range")
print("- KeyError: Dictionary key not found")
print("- FileNotFoundError: File doesn't exist")
print("- IOError: File I/O problems")
print("- AttributeError: Attribute doesn't exist")
print("- NameError: Variable doesn't exist")
print("=" * 60)



from fastapi import FastAPI
app=FastAPI()


@app.get("/")
def read_root():
    return {"message":"Hello World"}
    