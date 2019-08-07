import random
import string

def otpGenString (length=15,uppercase = True ,lowercase=True,numbers=True):
    character_set = ''

    if numbers:
        character_set+= string.digits
    if uppercase:
        character_set+= string.ascii_uppercase
    if lowercase:
        character_set+= string.ascii_lowercase

    return''.join(random.choice(character_set)for n in range(length))
