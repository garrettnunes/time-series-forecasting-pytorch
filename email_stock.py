import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.base import MIMEBase
from email import encoders
import os

class email():
    def __init__(self):
        self.mail_from = 'ai@fazer.tech'

    def send(self, mail_to, mail_subject, mail_htmlbody, ticker):
        limit = 0
        while limit < 3:
            try:
                limit += 1
                smtpsrv = "smtp.sendgrid.net"
                smtpserver = smtplib.SMTP(smtpsrv, 587)
                smtpserver.ehlo()
                smtpserver.starttls()
                smtpserver.login(
                    'apikey', os.environ['SENDGRID_KEY'])
                msg = MIMEMultipart('alternative')
                msg['Subject'] = mail_subject
                # msg['From'] = mail_from
                msg['From'] = formataddr(('Price Prediction', self.mail_from))
                msg['To'] = ','.join(mail_to)
                # HTMLformat = MIMEText(mail_htmlbody, 'plain')  # this is NOT HTML FORMAT
                HTMLformat = MIMEText(mail_htmlbody, 'html')
                msg.attach(HTMLformat)


                # Add the attachment
                for i in range(1, 6):
                    filedir = f"static//{ticker}_{i}.png"
                    filename = f"{ticker}_{i}.png"
                    with open(filedir, "rb") as attachment:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            "Content-Disposition",
                            f"attachment; filename= {filename}",
                        )
                        msg.attach(part)


                smtpserver.sendmail(self.mail_from, mail_to, msg.as_string())
                smtpserver.quit()
                return True
            except Exception as e:
                print(e)
        return False