         

import smtplib
import imaplib
import email
from email.message import EmailMessage


class EmailManager ():

    def __init__ (self, username, password):
        self._usrname = username
        self._pwd = password
        self._smtp_server= 'smtp.gmail.com'
        self._imap_server = "imap.gmail.com"
        self._imap_port = 993

    def send (self, to, subject, message):
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = self._usrname
        msg['To'] = ', '.join(to)
        msg.set_content(message)
        print(msg)
        server = smtplib.SMTP_SSL(self._smtp_server, 465)
        server.ehlo()
        server.set_debuglevel(0)
        server.login(self._usrname, self._pwd)
        server.send_message(msg)
        server.quit()

    def fetch_unread (self, sender_of_interest=None):
        # Login to INBOX
        imap = imaplib.IMAP4_SSL(self._imap_server, self._imap_port)
        imap.login(self._usrname, self._pwd)
        imap.select('INBOX')

        if sender_of_interest:
            status, response = imap.uid('search', None, 'UNSEEN', 'FROM {0}'.format(sender_of_interest))
        else:
            status, response = imap.uid('search', None, 'UNSEEN')
        #print (status, response)

        if status == 'OK':
            unread_msg_nums = response[0].split()
        else:
            unread_msg_nums = []
        data_list = []
        for e_id in unread_msg_nums:
            data_dict = {}
            e_id = e_id.decode('utf-8')
            _, response = imap.uid('fetch', e_id, '(RFC822)')
            html = response[0][1].decode('utf-8')
            email_message = email.message_from_string(html)
            data_dict['mail_to'] = email_message['To']
            data_dict['mail_subject'] = email_message['Subject']
            data_dict['mail_from'] = email.utils.parseaddr(email_message['From'])
            data_dict['body'] = email_message.get_payload()
            data_list.append(data_dict)
        
        imap.close()
        return data_list


class QPLmail (EmailManager):

    def __init__ (self, bay, password):
        username = 'qpl.bay'+str(bay)+'@gmail.com'
        super().__init__(username=username, password=password)

    def send (self, to, subject, message):
        subj = '[QPL-bay-'+self._bay+']: '+subject
        super().send(to=to, subject = subj, message = message)




