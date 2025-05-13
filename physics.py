import pygame
import numpy as np
import math as m
import matplotlib.pyplot as plt
from abc import *

pygame.init()

BLACK = (  0,  0,  0)
WHITE = (255,255,255)
RED   = (255,  0,  0)
GREEN = (  0,255,  0)
BLUE  = (  0,  0,255)

screen_size = np.array([1366,768])
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Physics")

clock = pygame.time.Clock()

object_list = []    #현재 있는 모든 moving object 객체들의 리스트
field_list = []     #현재 있는 모든 field 객체들의 리스트
prop_list = ["q"]   #moving object 객체가 가질 수 있는 추가 특성들의 key를 담은 리스트

def v_len(vector:np.ndarray):                   #주어진 벡터의 크기를 반환함
    return m.sqrt(vector.dot(vector))
def cross(a:np.ndarray,b:np.ndarray):           #두 2차원상 numpy 벡터의 외적의 크기를 반환함
    return a[0]*b[1]-a[1]*b[0]
def len_multiply(a:np.ndarray,b:np.ndarray):    #주어진 벡터의 크기의 곱을 반환
    return v_len(a)*v_len(b)
def polar(v:np.ndarray):                        #데카르트 좌표계 numpy 벡터를 극좌표계로 변환한 값을 반환
    x, y = v
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y, x)
    return np.array([r, th])
def cart(v:np.ndarray):                         #극좌표계 numpy 벡터를 데카르트 좌표계로 변환한 값을 반환
    r, th = v
    x = r * np.cos(th)
    y = r * np.sin(th)
    return np.array([x, y])
def element(v:np.ndarray,n:int):                #numpy 벡터를 받아 n번째 성분을 반환
    return v[n]
def dir(start:np.ndarray,end:np.ndarray):       #시작점과 끝점의 위치벡터를 받아 그 방향의 단위벡터를 반환
    return (end - start) / v_len(end - start)

def list_macro(arr:list,func):                  #주어진 리스트의 모든 내용에 대해 주어진 함수를 적용한 값을 반환
    return [func(i) for i in arr]
def integrate_list(list1:list,list2:list,func): #두 리스트의 값을 주어진 함수로 병합
    if len(list1) != len(list2):
        raise ValueError(f"""Two given list has different length
                             list1:{len(list1)} list2:{len(list2)}""")
    return [func(list1[i], list2[i]) for i in range(len(list1))]

#field 객체와 moving object 객체가 따로 있음.
#moving object는 field를 가질 수도, 아닐 수도 있음
#field는 독립적으로 존재할 수 있음

class moving_object():                      #움직일 수 있는 물체를 나타내는 객체
    def __init__(self,pos:list,v:list,m,color = RED,printV = False,printA = False,printHistory = 0,prop:list = [0]*len(prop_list)):

        self.prop = {}
        self.prop["pos"] = np.array(pos)    #물체의 위치
        self.prop["v"] = np.array(v)        #물체의 속도
        self.prop["a"] = np.array([0,0])    #물체의 가속도
        self.prop["m"] = m                  #물체의 질량

        for i,key in enumerate(prop_list):
            self.prop[key] = prop[i]        #물체가 가지는 추가 특성들의 딕셔너리
        
        self.printV = printV                #draw 함수 호출 시 속도 막대 출력 여부
        self.printA = printA                #draw 함수 호출 시 가속도 막대 출력 여부
        self.printHistory = printHistory    #draw 함수 호출 시 물체의 자취 곡선 출력 여부
        self.color = color                  #draw 함수 호출 시 물체의 출력 색상

        self.field_list = []                #객체에 종속된 장들의 리스트
        self.history = []                   #물체의 자취를 나타내는 리스트(최대 1000틱 저장)
        for i in range(10000):
            self.history.append(self.prop["pos"] + screen_size/2)
        object_list.append(self)            #물체의 리스트에 자신을 등록
        
    def draw(self):                         #pygame 화면 위에 물체를 출력
        pygame.draw.circle(screen,self.color,self.prop["pos"] + screen_size/2,5)
        if self.printV:
            pygame.draw.line(screen,self.color,self.prop["pos"] + screen_size/2,self.prop["pos"] + screen_size/2 + self.prop["v"]/10,3)
        if self.printA:
            pygame.draw.line(screen,self.color,self.prop["pos"] + screen_size/2,self.prop["pos"] + screen_size/2 + self.prop["a"]/20,6)
        if self.printHistory > 0 and self.printHistory <= 10000:
            pygame.draw.lines(screen,self.color,False,self.history[-self.printHistory:])
    def set_a(self):                        #물체의 현재 가속도를 계산하고 자신의 가속도 변수에 저장함
        self.prop["a"] = 0
        Field:field
        for Field in field_list:
            if Field.parent != self:
                self.prop["a"] += Field.GetAcc(self)

class field(metaclass = ABCMeta):                       #하나의 장을 나타내는 추상 클래스
    @abstractmethod
    def __init__(self,parent:moving_object = 0,k=1):    #k는 장의 힘 상수, parent(종속된 moving object 객체)가 있을 시 다른 매개변수들은 모두 무시됨
        self.k = k
        self.parent = parent
        field_list.append(self)
        if parent == 0: pass
        else: parent.field_list.append(self)

    @abstractmethod
    def GetForce(self,opponent:moving_object):          #어떤 moving object 객체가 이 field 객체에 의해 받는 힘을 반환
        pass
    @abstractmethod
    def refresh_var(self):                              #이 객체가 가진 값이 상위 객체(parent)의 변수에 의존할 때 사용
        pass
    def GetAcc(self,opponent:moving_object):            #어떤 moving object 객체가 이 field 객체에 의해 받을 가속도를 반환
        return self.GetForce(opponent)/opponent.prop["m"] 

class central_force(field, metaclass = ABCMeta):                            #장 중에서 중심력장을 나타내는 추상 클래스
    @abstractmethod
    def __init__(self,parent:moving_object = 0,k=1,center:list = [0,0]):    #center는 중심력장의 중심
        if parent == 0:
            super().__init__(k=k)
            self.center = np.array(center)
        else:
            super().__init__(parent,k=k)
            self.center = self.parent.prop["pos"]

    @abstractmethod
    def GetForce(self,opponent):
        pass
    def refresh_var(self):
        if self.parent != 0:
            self.center = self.parent.prop["pos"]

class gravity_field(central_force):                                         #중력장을 나타내는 클래스
    def __init__(self,parent:moving_object = 0,m=1,center:list = [0,0]):    #m은 중력의 원인의 질량
        if parent == 0:
            super().__init__(center=center,k=5000*m)
            self.m = m
        else:
            super().__init__(parent,k=5000*parent.prop["m"])
            self.m = self.parent.prop["m"]

    def GetForce(self,opponent:moving_object):                              #F=k*m1*m2/r^2 - 중력의 크기를 나타내는 식
        return (dir(opponent.prop["pos"],self.center) * 
                self.k * self.m * opponent.prop["m"] / 
                (opponent.prop["pos"]-self.center).dot(opponent.prop["pos"]-self.center))
    
class record():     #원하는 물리량을 계속해서 기록하고 matplotlib 플롯으로 나타내주는 클래스
    def __init__(self):
        self.DATA_address = {}  #저장한 데이터들의 위치를 담은 딕셔너리
        self.DATA         = {}  #데이터들이 담긴 딕셔너리
        self.DATA_t= [time]     #시간 리스트

    def add_data(self,obj:moving_object,key:str,name:str):  #기록할 데이터 추가
        self.DATA_address[name] = (obj,key)
        self.DATA[name] = [obj.prop[key]]
    def record_data(self):                                  #기록할 데이터들을 DATA 딕셔너리에 기록
        data:list
        for name,data in self.DATA.items():
            data.append(self.DATA_address[name][0].prop[self.DATA_address[name][1]])
        self.DATA_t.append(time)

class graph:        #그래프 생성을 돕는 함수들의 집합
    def polar_eclipse2(self,a,b): #데카르트 좌표계 그래프 x^2/a^2 + x^2/b^2 = 1를 가지고 극방정식 작성
        pass

#######################################################################################################

#케플러 법칙 탐구
obj1 = moving_object([400,0],[0,500],1,RED,True,True,10000)
field1 = gravity_field(m=200)

#이체 문제1
#obj1 = moving_object([0,0],[0,-2],200,RED,True,True,100)
#gravity1 = gravity_field(parent = obj1)
#obj2 = moving_object([400,0],[0,300],1,BLUE,True,True,100)
#gravity2 = gravity_field(parent = obj2)

#이체 문제2 - 정확도 향상 필요
#obj1 = moving_object([100,0],[0,300],100,RED,True,True,500)
#gravity1 = gravity_field(parent = obj1)
#obj2 = moving_object([-100,0],[0,-300],100,BLUE,True,True,500)
#gravity1 = gravity_field(parent = obj2)

#삼체 문제 - 변수 조정 필요
#dt = 0.008
#obj1 = moving_object([-200,0],[0,-10],13,RED,True,True,1000)
#gravity1 = gravity_field(parent = obj1)
#obj2 = moving_object([20,0],[0,87],13,BLUE,True,True,1000)
#gravity2 = gravity_field(parent = obj2)
#obj3 = moving_object([168,0],[0,-77],13,GREEN,True,True,1000)
#gravity3 = gravity_field(parent = obj3)

########################################################################################################

dt = 0.0003
time = dt
recorder = record()
recorder.add_data(obj1,"pos","obj1_r")
recorder.add_data(obj1,"v","obj1_v")

running = True
while running:

    tick = clock.tick()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    screen.fill(WHITE)
    for i in range(100):
        obj:moving_object
        for obj in object_list:
            obj.set_a()
            obj.prop["v"] = obj.prop["v"] + obj.prop["a"] * (dt/100)
            obj.prop["pos"] = obj.prop["pos"] + obj.prop["v"] * (dt/100)
    for obj in object_list:
        obj.history = obj.history[1:]
        obj.history.append(obj.prop["pos"] + screen_size/2)
        obj.draw()
        Field:field
    for Field in field_list:
        Field.refresh_var()
    recorder.record_data()

    time += dt
    pygame.display.update()

plt.rcParams['font.family'] = 'NanumGothic'
plt.title("Conservation of Angular Momentum in Planet Orbit (30408 김태언)")
plt.xlabel("time[sec]")
plt.plot(recorder.DATA_t,integrate_list(recorder.DATA["obj1_r"],recorder.DATA["obj1_v"],cross),label = "r x v")
plt.plot(recorder.DATA_t,integrate_list(recorder.DATA["obj1_r"],recorder.DATA["obj1_v"],len_multiply),label = "|r|*|v|")
plt.axis([0,time,0,400000])
plt.legend()

pygame.quit()
plt.show()
