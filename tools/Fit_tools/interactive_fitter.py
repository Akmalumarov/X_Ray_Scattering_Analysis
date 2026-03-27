import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets, VBox, HBox, Layout
from IPython.display import display
from scipy.optimize import curve_fit
from collections import OrderedDict
import inspect

class Fitter:
    def __init__(self, x, y, lib, x0_limits=None):
        self.x = x
        self.y = y
        self.lib = lib
        self.x0_limits = x0_limits
        self.comps = OrderedDict()
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        self.setup_ui()
        
    def setup_ui(self):
        self.selector = widgets.Dropdown(options=list(self.lib.keys()), description='Функция:')
        self.add_btn = widgets.Button(description='Добавить', button_style='success')
        self.add_btn.on_click(self.add)
        self.fit_btn = widgets.Button(description='Фитить', button_style='primary')
        self.fit_btn.on_click(self.fit)
        
        self.comps_box = widgets.VBox([])
        self.out = widgets.Output()
        
        display(VBox([HBox([self.selector, self.add_btn, self.fit_btn]), self.comps_box, self.out]))
        self.plot()
        
    def add(self, _):
        name = self.selector.value
        func = self.lib[name]
        
        sig = inspect.signature(func)
        params = [p for p in sig.parameters if p != 'x']
        
        widgs = []
        for p in params:
            w = widgets.FloatText(value=1.0, description=f'{p}:', layout=Layout(width='150px'), step=0.1)
            w.observe(self.plot, 'value')
            widgs.append(w)
        
        cb = widgets.Checkbox(value=True, description='показать', layout=Layout(width='90px'))
        cb.observe(self.plot, 'value')
        
        rem = widgets.Button(description='✖', layout=Layout(width='40px'), button_style='danger')
        
        comp_id = len(self.comps)
        self.comps[comp_id] = {
            'name': name, 'func': func, 'params': widgs, 'enabled': cb,
            'color': self.colors[comp_id % len(self.colors)]
        }
        
        rem.on_click(lambda _, cid=comp_id: self.remove(cid))
        
        self.comps_box.children += (HBox([widgets.HTML(f"<b>{name}</b>", layout=Layout(width='100px'))] + widgs + [cb, rem]),)
        self.plot()
        
    def remove(self, cid):
        del self.comps[cid]
        children = []
        for cid, c in self.comps.items():
            rem = widgets.Button(description='✖', layout=Layout(width='40px'), button_style='danger')
            rem.on_click(lambda _, cid=cid: self.remove(cid))
            children.append(HBox([widgets.HTML(f"<b>{c['name']}</b>", layout=Layout(width='100px'))] + 
                                c['params'] + [c['enabled'], rem]))
        self.comps_box.children = tuple(children)
        self.plot()
        
    def model(self, x, *params):
        y = np.zeros_like(x)
        idx = 0
        for c in self.comps.values():
            if c['enabled'].value:
                n = len(c['params'])
                y += c['func'](x, *params[idx:idx+n])
                idx += n
        return y
        
    def plot(self, _=None):
        self.out.clear_output(wait=True)
        with self.out:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            ax1.plot(self.x, self.y, 'k.', markersize=2, label='data')
            x_smooth = np.linspace(self.x.min(), self.x.max(), 1000)
            
            y_total = np.zeros_like(self.x)
            y_total_smooth = np.zeros_like(x_smooth)
            
            for c in self.comps.values():
                if c['enabled'].value:
                    p = [w.value for w in c['params']]
                    yc = c['func'](self.x, *p)
                    y_total += yc
                    yc_smooth = c['func'](x_smooth, *p)
                    y_total_smooth += yc_smooth
                    ax1.plot(x_smooth, yc_smooth, '--', color=c['color'], alpha=0.7, label=c['name'])
            
            ax1.plot(x_smooth, y_total_smooth, 'r-', linewidth=2, label='sum')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(self.x, self.y - y_total, 'b-')
            ax2.axhline(0, color='k', ls='--', lw=0.5)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlabel('x')
            
            plt.tight_layout()
            plt.show()
            
    def fit(self, _):
        p0 = []
        bounds_low = []
        bounds_high = []
        
        for c in self.comps.values():
            if c['enabled'].value:
                for w in c['params']:
                    p0.append(w.value)
                    param_name = w.description.replace(':', '')
                    
                    if param_name == 'x0' and self.x0_limits:
                        bounds_low.append(self.x0_limits[0])
                        bounds_high.append(self.x0_limits[1])
                    else:
                        bounds_low.append(0)
                        bounds_high.append(np.inf)
        
        if not p0:
            return
            
        try:
            popt, _ = curve_fit(self.model, self.x, self.y, p0=p0, bounds=(bounds_low, bounds_high))
        except Exception as e:
            print(f"Ошибка фита: {e}")
            return
        
        idx = 0
        for c in self.comps.values():
            if c['enabled'].value:
                n = len(c['params'])
                for j, w in enumerate(c['params']):
                    w.value = popt[idx + j]
                idx += n
        
        self.plot()

    def integrals(self):
        """Возвращает площади для каждой компоненты"""
        areas = []
        for c in self.comps.values():
            if c['enabled'].value:
                p = [w.value for w in c['params']]
                yc = c['func'](self.x, *p)
                areas.append(np.trapezoid(yc, self.x))
        return areas
    
    def moments(self):
        """Возвращает моменты (интегралы с весом x^2) для каждой компоненты"""
        moments = []
        for c in self.comps.values():
            if c['enabled'].value:
                p = [w.value for w in c['params']]
                yc = c['func'](self.x, *p)
                moment = np.trapezoid(yc * self.x**2, self.x)
                moments.append(moment)
        return moments
    
    def all_integrals(self):
        """Возвращает оба типа интегралов для каждой компоненты"""
        results = []
        for c in self.comps.values():
            if c['enabled'].value:
                p = [w.value for w in c['params']]
                yc = c['func'](self.x, *p)
                area = np.trapezoid(yc, self.x)
                moment = np.trapezoid(yc * self.x**2, self.x)
                results.append({
                    'name': c['name'],
                    'area': area,
                    'moment': moment
                })
        return results

    def get_params(self):
        params = []
        for c in self.comps.values():
            if c['enabled'].value:
                params.append({
                    'name': c['name'],
                    'params': [w.value for w in c['params']]
                })
        return params
    
    def set_params(self, params):
        for p in params:
            self.selector.value = p['name']
            self.add(None)
            last_id = len(self.comps) - 1
            for w, val in zip(self.comps[last_id]['params'], p['params']):
                w.value = val
        self.plot()

    def get_model(self):
        def model_func(x, *params):
            y = np.zeros_like(x)
            idx = 0
            for c in self.comps.values():
                if c['enabled'].value:
                    n = len(c['params'])
                    y += c['func'](x, *params[idx:idx+n])
                    idx += n
            return y
        
        p0 = []
        for c in self.comps.values():
            if c['enabled'].value:
                p0 += [w.value for w in c['params']]
        
        return model_func, p0