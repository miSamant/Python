from otpGen import otpGenString

count = -1
Otp=otpGenString(5)
print(Otp)
check=''
print('Enter otp')
check=input()

if check==Otp:
    print('Success')
else:
    print('Fail')
