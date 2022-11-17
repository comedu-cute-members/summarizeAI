import tkinter as tk
from tkinter import filedialog
import time

#Load button command
def LoadButtonPress():
    OriginalTextDirectory = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("TXT files", "*.txt"),
                                          ("all files", "*.*")))
    FileLoad=open(OriginalTextDirectory,'r')
    OriginalText=FileLoad.read()
    OriginalTextBox.insert(tk.END, OriginalText)
    FileLoad.close()

#Summary button command
def SummaryButtonPress():
    
    return
    
#Save button command
def SaveButtonPress():
    SummaryTextDirectory = filedialog.asksaveasfilename(initialdir="/", title="Select file",
                                          filetypes=(("TXT files", "*.TXT"),
                                          ("all files", "*.*")))
    FileSave=open(SummaryTextDirectory,'w', encoding='utf-8')
    SummaryText=SummaryTextBox.get(1.0, "end")
    FileSave.write(SummaryText)
    FileSave.close()

    #파일 저장 시간 표시
    now = time.localtime()
    SaveTime = str(now.tm_year)+'/'+str(now.tm_mon)+'/'+str(now.tm_mday)+' '+str(now.tm_hour)+':'+str(now.tm_min)+':'+str(now.tm_sec)
    SaveTimeLabel = tk.Label(SummaryAI, text=SaveTime+' save complete', font=('Arial',10))
    SaveTimeLabel.place(x=1100, y=720)
    SaveTimeLabel.configure(bg='white')

    #파일 실행 프로그램 안내
    TextFileNotice = tk.Tk()
    TextFileNotice.geometry("400x150")
    TextFileNotice.title("TextFileNotice")
    TextNote=tk.Label(TextFileNotice, text="생성된 파일을 \n메모장으로 실행하세요", font=('Arial',15))
    TextNote.pack()
    
    ConfirmButton=tk.Button(TextFileNotice, text='확인')
    ConfirmButton.place(x=160, y=100)
    ConfirmButton.config(width=10)
    ConfirmButton.config(command=TextFileNotice.destroy)
    
    TextFileNotice.mainloop()

#Help button command
def HelpButtonPress():
    HelpExplain = tk.Tk()
    HelpExplain.geometry("700x400")
    HelpExplain.title("Help")
    HelpExplainLoad=tk.Label(HelpExplain, text="Load: 저장된 텍스트 파일을 풀러오세요.\n\n", font=('Arial',15))
    HelpExplainSummary=tk.Label(HelpExplain, text="Summary: 원문을 요약하려면 Summary버튼을 클릭하세요.\n\n", font=('Arial',15))
    HelpExplainSave=tk.Label(HelpExplain, text="Save: 요약된 텍스트 파일을 원하는 경로에 저장하세요. \n\n", font=('Arial',15))
    HelpExplainQuit=tk.Label(HelpExplain, text="Quit: 프로그램을 종료합니다.", font=('Arial',15))

    HelpExplainLoad.pack()
    HelpExplainSummary.pack()
    HelpExplainSave.pack()
    HelpExplainQuit.pack()



#gui생성
SummaryAI = tk.Tk()
SummaryAI.geometry("1920x1080")
SummaryAI.configure(bg='white')
SummaryAI.title("SummaryAI")

#Text 상자 생성
OriginalTextBox = tk.Text(SummaryAI)
OriginalTextBox.configure(bg='white')
SummaryTextBox = tk.Text(SummaryAI)
SummaryTextBox.configure(bg='white')

#상단 제목 작성
SummaryAITitle = tk.Label(SummaryAI, text="SummaryAI", font=('Arial', 40))
SummaryAITitle.place(x=610, y=30)
SummaryAITitle.configure(bg='white')

#Text 상자 위치 조정
OriginalTextBoxMessage = tk.Label(SummaryAI, text="요약할 원문을 붙여넣기 하거나 파일을 불러오세요.", font=('Arial', 15))
OriginalTextBoxMessage.place(x=190, y=110)
OriginalTextBoxMessage.configure(bg='white')
OriginalTextBox.grid(row = 0, column = 0, padx = 120, pady = 150, ipadx = 0, ipady = 120)

SummaryTextBoxMessage = tk.Label(SummaryAI, text="요약된 텍스트가 여기 표시됩니다.", font=('Arial', 15))
SummaryTextBoxMessage.place(x=973, y=110)
SummaryTextBoxMessage.configure(bg='white')
SummaryTextBox.grid(row = 0, column = 1, padx = 20, pady = 150, ipadx = 0, ipady = 120)


#화살표 이미지 삽입
ArrowImage=tk.PhotoImage(file="화살표.png", master=SummaryAI)
ArrowLabel=tk.Label(SummaryAI, image=ArrowImage)
ArrowLabel.configure(bg='white')
ArrowLabel.place(x=700, y=350)

#LoadButton 생성
LoadButton = tk.Button(SummaryAI, text="Load")
LoadButton.place(x=119, y=720)
LoadButton.config(width=10, bg='orange')
LoadButton.config(command=LoadButtonPress)
    
#SummaryButton 생성
SummaryButton = tk.Button(SummaryAI, text="Summary")
SummaryButton.place(x=709, y=500)
SummaryButton.config(width=12, height=2, bg='skyblue')
SummaryButton.config(command=SummaryButtonPress)

#SaveButton 생성
SaveButton = tk.Button(SummaryAI, text="Save")
SaveButton.place(x=1310, y=720)
SaveButton.config(width=10, bg='orange')
SaveButton.config(command=SaveButtonPress)

#HelpButton 생성
HelpButton = tk.Button(SummaryAI, text="Help")
HelpButton.place(x=119, y=30)
HelpButton.config(width=10, bg='gray')
HelpButton.config(command=HelpButtonPress)

#QuitButton 생성
QuitButton = tk.Button(SummaryAI, text="Quit")
QuitButton.place(x=1310, y=755)
QuitButton.config(width=10, bg='red')
QuitButton.config(command=SummaryAI.destroy)



SummaryAI.mainloop()



