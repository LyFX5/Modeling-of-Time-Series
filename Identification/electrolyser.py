
import numpy as np
import matplotlib.pyplot as plt

class Electrolyser(): # базовый динамический объект родитель всех динамических объектов в проекте
    def __init__(self, ID, delta_t):
        self.ID = ID

        self.state = 'idle'
        self.target = 0

        self.y_max = 0

        self.envTemperature = 30.0
        self.maxTemperature = 54.2
        [Temper, dTemper] = [self.envTemperature, 0.]
        self.TemperatureDinamics = [Temper, dTemper]

        #self.Temperature = 0 # на данный момент dT = self.dTemper*dt, а потом останавливается

        self.delta_t = delta_t
        #self.dTemper = 0.2*self.delta_t

        [y, yd, ydd] = [0.,0.,0.]
        self.dinamics = [y, yd, ydd]

        self.u_log = []

        # TODO other parameters

        '''
        Mparams = [Temperature, Voltage, Degrad, State, prod_rate_value, consuming_power]
        self.cost_of_switchOn = 1
        self.cost_of_switchOff = 1
        self.cost_of_work_for_a_minute = 1

        self.run_out = 0
        '''


    def getDinamics(self):
        return self.dinamics

    def updateDinamics(self, newDinamics, newTemperDinamics):
        [y, yd, ydd] = newDinamics
        self.dinamics[0] = y
        self.dinamics[1] = yd
        self.dinamics[2] = ydd

        [Temper, Temper_d] = newTemperDinamics
        self.TemperatureDinamics[0] = Temper
        self.TemperatureDinamics[1] = Temper_d

    def getTemperatureDinamics(self):
        return self.TemperatureDinamics

    def apply_control_signal_in_moment(self, U): # integration method
        assert 60 <= U <= 100 or U == 0

        [y_prev, yd_prev, ydd_prev] = self.getDinamics()

        [Temperature_prev, Temperature_d_prev] = self.getTemperatureDinamics()

        if self.state == 'idle':
            self.target = U/100.

            ud = 0
            yd = 0
            y = 0
            ydd = 0

            ## Temperature modeling
            # in idle state just cooling

            T_Temper_idle = 1000
            Temper_ctr_coef = 1

            Temperature = Temperature_prev + self.delta_t * Temperature_d_prev
            Temperature_d = -Temperature / T_Temper_idle + Temper_ctr_coef * self.envTemperature / T_Temper_idle

            if self.target != 0:
                self.state = 'hydration'

            self.u_log.append(0)

        elif self.state == 'hydration':
            ud = 0
            yd = 0
            y = 0
            ydd = 0

            ## Temperature modeling
            # in hydration state heating

            T_Temper_hydration = 700 # 3726.0
            Temper_ctr_coef_hydration = 1

            Temperature = Temperature_prev + self.delta_t * Temperature_d_prev
            Temperature_d = self.maxTemperature / T_Temper_hydration

            if True:#Temperature >= 34.6: # ============================================= FIT
                self.state = 'ramp_up_1'

            self.u_log.append(0)

        elif self.state == 'ramp_up_1':

            if y_prev == 0:
                u_cur = self.target
                u_prev = self.u_log[-1]
                ud = (u_cur - u_prev) / self.delta_t
                yd = yd_prev + self.delta_t * ydd_prev
                y = 0.08*self.target # ============================================= FIT

            else:
                u_cur = self.target
                u_prev = self.u_log[-1]
                ud = (u_cur - u_prev) / self.delta_t
                yd = yd_prev + self.delta_t * ydd_prev
                y = y_prev + self.delta_t * yd_prev

            # меньше коэффициенты знаменателя => меньше скорость # ============================================= FIT
            a0 = 0.000225 #0.000004 #0.09 # 0.01 #9 #0.01 #0.25  # 1
            a1 = 0.03 #0.004 #0.6 #0.2 #1  # 2
            b0 = 1*a0 #* 0.25
            b1 = -22*a0 #9 #* 0.25
            ydd = -a0 * y - a1 * yd + b1 * ud + b0 * u_cur

            ## Temperature modeling
            # in ramp_up_1 state

            T_Temper_ramp_up_1 = 3500 # T = t_max / ((u_vals[0] - y_vals[0]) / u_vals[0])
            Temper_ctr_coef_ramp_up_1 = 2

            Temperature = Temperature_prev + self.delta_t * Temperature_d_prev
            Temperature_d = self.maxTemperature / T_Temper_ramp_up_1

            y_targ_in_ramp_up_1 = self.target * 0.11 # ============================================= FIT

            if y >= y_targ_in_ramp_up_1:
                self.state = 'ramp_up_2'

            self.u_log.append(u_cur)

        elif self.state == 'ramp_up_2':

            T_ramp_up_2 = 500 # ============================================= FIT
            # T_ramp_up_2 = f(Temperature) тем больше (то есть процес тем медленнее, чем меньше температура)
            ctr_coef = 1.097 # ============================================= FIT

            ud = 0
            u_cur = self.target
            #yd = u_cur/T_ramp_up_2
            y = y_prev + self.delta_t * yd_prev
            yd = -y / T_ramp_up_2 + ctr_coef * u_cur / T_ramp_up_2
            ydd = 0

            ## Temperature modeling
            # in ramp_up_2 state

            T_Temper_ramp_up_2 = 3400 # 3726.0
            Temper_ctr_coef_ramp_up_2 = 2

            Temperature = Temperature_prev + self.delta_t * Temperature_d_prev
            Temperature_d = self.maxTemperature / T_Temper_ramp_up_2 # температура при рампапах растет неограниченно пока ток не выйдет в steady

            if y >= self.target:
                self.state = 'steady'

            self.u_log.append(u_cur)

        elif self.state == 'steady':

            self.target = U / 100.
            u_cur = self.target

            T_steady = 55 # ============================================= FIT

            ud = 0
            #yd = -y_prev/T_steady + u_cur/T_steady
            y = y_prev + self.delta_t * yd_prev
            yd = -y / T_steady + u_cur / T_steady
            ydd = 0

            # if self.u_log[-1] == u_cur:
            #     y = y_prev + self.delta_t * yd_prev
            #     yd = 0 #-y / T_steady + u_cur / T_steady
            #
            # else:
            #     self.y_steady = y
            #     y = y_prev + self.delta_t * yd_prev
            #     yd = -y / T_steady + u_cur / T_steady
            #

            # замоделировать снижение и повышение уровня, потому что это происходит не по такому же закону

            ## Temperature modeling
            # in steady state

            T_Temper_steady = 200
            Temper_ctr_coef_steady = 1 # тут всегда должен быть 1 потому что это steady

            Temperature = Temperature_prev + self.delta_t * Temperature_d_prev
            Temperature_d = -Temperature / T_Temper_steady + Temper_ctr_coef_steady * self.maxTemperature / T_Temper_steady #50 / T_Temper_steady
            # умножаю self.maxTemperature*(y) чтобы температура понижалась при понижении тока (0 <= y <= 1)

            if self.target == 0:
                self.state = 'ramp_down_1'
                self.y_max = y

            self.u_log.append(u_cur)

        elif self.state == 'ramp_down_1':

            T_ramp_down_1 = 20 # ============================================= FIT

            ud = 0
            u_cur = 0 # = self.target
            yd = -self.y_max / T_ramp_down_1 + u_cur / T_ramp_down_1
            y = y_prev + self.delta_t * yd
            ydd = 0

            ## Temperature modeling
            # in ramp_down_1 state

            T_Temper_ramp_down_1 = 1000
            Temper_ctr_coef_steady = 1

            Temperature = Temperature_prev + self.delta_t * Temperature_d_prev
            Temperature_d = -500 / T_Temper_ramp_down_1

            if y_prev <= 0.5: # ============================================= FIT
                self.state = 'ramp_down_2'

            self.u_log.append(u_cur)

        elif self.state == 'ramp_down_2':

            T_ramp_down_2 = 30 # ============================================= FIT

            ud = 0
            u_cur = 0 # = self.target
            yd = -self.y_max / T_ramp_down_2 + u_cur / T_ramp_down_2
            y = y_prev + self.delta_t * yd
            ydd = 0

            ## Temperature modeling
            # in ramp_down_2

            T_Temper_ramp_down_2 = 1000
            Temper_ctr_coef_steady = 1

            Temperature = Temperature_prev + self.delta_t * Temperature_d_prev
            Temperature_d = -500 / T_Temper_ramp_down_2

            if y_prev <= 0.0001: # ============================================= FIT ?
                self.state = 'idle'

            self.u_log.append(u_cur)

        else:
            print("wrong state")

        newDinamics = [y, yd, ydd]
        newTemperDinamics = [Temperature, Temperature_d]
        self.updateDinamics(newDinamics, newTemperDinamics)

#================================================================ TEST MODELING ELECTROLYSER
'''
delta_t = 1 # time step size (seconds)

t_max = 2000

time_work = np.linspace(0, t_max, int(t_max//delta_t))

y_vals = [] # 0.2
yd_vals = []
ydd_vals = []


# ud_vals = []

Temperature = []

sys = Electrolyser(1, delta_t)

print(sys.state)

U = 100.
u_vals = (U/100.)*np.ones_like(time_work)

for i in range(len(time_work)): # [0:int(15/delta_t)]

    sys.apply_control_signal_in_moment( U )
    [y, yd, ydd] = sys.getDinamics()

    #ud_vals.append(ud)
    y_vals.append(y)
    yd_vals.append(yd)
    ydd_vals.append(ydd)

    Temperature.append(sys.Temperature)

print(sys.state)
'''

'''
U = 0
u_vals = (U/100.)*np.ones_like(time_work)

for i in range(len(time_work[int(15/delta_t):])):

    sys.apply_control_signal_in_moment( U )
    [y, yd, ydd] = sys.getDinamics()

    #ud_vals.append(ud)
    y_vals.append(y)
    yd_vals.append(yd)
    ydd_vals.append(ydd)

    Temperature.append(sys.Temperature)

print(sys.state)
'''

'''
plt.title("Step response y(t)")
plt.plot(time_work, u_vals, label='u')
plt.plot(time_work, y_vals, label='y')
#plt.plot(time_work, Temperature, label='T')
plt.legend()
plt.grid(visible=True)
plt.show()
'''