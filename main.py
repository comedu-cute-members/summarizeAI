import tkinter as tk
from tkinter import filedialog
import time

#load_button_command
def load_button_press():
    original_text_directory = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("TXT files", "*.txt"),
                                          ("all files", "*.*")))
    file_load=open(original_text_directory,'r')
    original_text=file_load.read()
    original_text_box.insert(tk.END, original_text)
    file_load.close()

#summary_button_command
def summary_button_press():
    
    return
    
#save_button_command
def save_button_press():
    summary_text_directory = filedialog.asksaveasfilename(initialdir="/", title="Select file",
                                          filetypes=(("TXT files", "*.TXT"),
                                          ("all files", "*.*")))
    file_save=open(summary_text_directory,'w', encoding='utf-8')
    summary_text=summary_text_box.get(1.0, "end")
    file_save.write(summary_text)
    file_save.close()

    #파일 저장 시간 표시
    now = time.localtime()
    save_time = str(now.tm_year)+'/'+str(now.tm_mon)+'/'+str(now.tm_mday)+' '+str(now.tm_hour)+':'+str(now.tm_min)+':'+str(now.tm_sec)
    save_time_label = tk.Label(summary_AI, text=save_time+' save complete', font=('Arial',10))
    save_time_label.place(x=1100, y=720)
    save_time_label.configure(bg='white')

    #파일 실행 프로그램 안내
    text_file_notice = tk.Tk()
    text_file_notice.geometry("400x150")
    text_file_notice.title("TextFileNotice")
    text_note=tk.Label(text_file_notice, text="생성된 파일을 \n메모장으로 실행하세요", font=('Arial',15))
    text_note.pack()
    
    confirm_button=tk.Button(text_file_notice, text='확인')
    confirm_button.place(x=160, y=100)
    confirm_button.config(width=10)
    confirm_button.config(command=text_file_notice.destroy)
    
    text_file_notice.mainloop()

#help_button_command
def help_button_press():
    help_explain = tk.Tk()
    help_explain.geometry("700x400")
    help_explain.title("Help")
    help_explain_load = tk.Label(help_explain, text="Load: 저장된 텍스트 파일을 풀러오세요.\n\n", font=('Arial',15))
    help_explain_summary = tk.Label(help_explain, text="Summary: 원문을 요약하려면 Summary버튼을 클릭하세요.\n\n", font=('Arial',15))
    help_explain_save = tk.Label(help_explain, text="Save: 요약된 텍스트 파일을 원하는 경로에 저장하세요. \n\n", font=('Arial',15))
    help_explain_quit = tk.Label(help_explain, text="Quit: 프로그램을 종료합니다.", font=('Arial',15))

    help_explain_load.pack()
    help_explain_summary.pack()
    help_explain_save.pack()
    help_explain_quit.pack()



#gui생성
summary_AI = tk.Tk()
summary_AI.geometry("1920x1080")
summary_AI.configure(bg='white')
summary_AI.title("SummaryAI")

#Text 상자 생성
original_text_box = tk.Text(summary_AI)
original_text_box.configure(bg='lightgray')
summary_text_box = tk.Text(summary_AI)
summary_text_box.configure(bg='lightgray')

#상단 제목 작성
summary_AI_title = tk.Label(summary_AI, text="SummaryAI", font=('Arial', 40))
summary_AI_title.place(x=610, y=30)
summary_AI_title.configure(bg='white')

#Text 상자 위치 조정
original_text_box_message = tk.Label(summary_AI, text="요약할 원문을 붙여넣기 하거나 파일을 불러오세요.", font=('Arial', 15))
original_text_box_message.place(x=190, y=110)
original_text_box_message.configure(bg='white')
original_text_box.grid(row = 0, column = 0, padx = 120, pady = 150, ipadx = 0, ipady = 120)

summary_text_box_message = tk.Label(summary_AI, text="요약된 텍스트가 여기 표시됩니다.", font=('Arial', 15))
summary_text_box_message.place(x=973, y=110)
summary_text_box_message.configure(bg='white')
summary_text_box.grid(row = 0, column = 1, padx = 20, pady = 150, ipadx = 0, ipady = 120)


#화살표 이미지 삽입
arrow_image=tk.PhotoImage(file="arrowimage.png", master=summary_AI)
arrow_label=tk.Label(summary_AI, image=arrow_image)
arrow_label.configure(bg='white')
arrow_label.place(x=700, y=350)

#load_button 생성
load_button = tk.Button(summary_AI, text="Load")
load_button.place(x=119, y=720)
load_button.config(width=10, bg='skyblue')
load_button.config(command=load_button_press)
    
#summary_button 생성
summary_button = tk.Button(summary_AI, text="Summary")
summary_button.place(x=709, y=500)
summary_button.config(width=12, height=2, bg='skyblue')
summary_button.config(command=summary_button_press)

#save_button 생성
save_button = tk.Button(summary_AI, text="Save")
save_button.place(x=1310, y=720)
save_button.config(width=10, bg='skyblue')
save_button.config(command=save_button_press)

#help_button 생성
help_button = tk.Button(summary_AI, text="Help")
help_button.place(x=119, y=755)
help_button.config(width=10, bg='skyblue')
help_button.config(command=help_button_press)

#quit_button 생성
quit_button = tk.Button(summary_AI, text="Quit")
quit_button.place(x=1310, y=755)
quit_button.config(width=10, bg='skyblue')
quit_button.config(command=summary_AI.destroy)



summary_AI.mainloop()
