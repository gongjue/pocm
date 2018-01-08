import os

def send_message(text):

     os.system("""osascript -e '
        tell application "Messages"
        send "%s" to buddy "dukegj@me.com" of (service 1 whose service type is iMessage)
        end tell'
        """ % text)


def main():
    send_message("jue")


if __name__ == "__main__":
    main()

