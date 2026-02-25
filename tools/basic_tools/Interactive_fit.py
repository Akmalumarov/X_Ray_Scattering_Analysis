import numpy as np
import plotly.graph_objects as go
from scipy import optimize
import ipywidgets as widgets
from IPython.display import display

def create_interactive_fitter(x, y, title="Интерактивный фит", width=1200, height=700):
    
    # Преобразуем в numpy массивы
    x = np.array(x)
    y = np.array(y)
    
    # Начальные значения
    x_min = float(min(x))
    x_max = float(max(x))
    x_range = x_max - x_min
    
    # ============= БИБЛИОТЕКА ФУНКЦИЙ =============
    
    # Словарь с функциями: название -> (функция, количество параметров, начальные приближения)
    functions = {
        # Линейные и полиномиальные
        'linear': {
            'func': lambda x, k, b: k * x + b,
            'n_params': 2,
            'params_names': ['k', 'b'],
            'p0': [1.0, 0.0],
            'description': 'k*x + b'
        },
        'quadratic': {
            'func': lambda x, a, b, c: a * x**2 + b * x + c,
            'n_params': 3,
            'params_names': ['a', 'b', 'c'],
            'p0': [1.0, 1.0, 0.0],
            'description': 'a*x² + b*x + c'
        },
        'cubic': {
            'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
            'n_params': 4,
            'params_names': ['a', 'b', 'c', 'd'],
            'p0': [1.0, 1.0, 1.0, 0.0],
            'description': 'a*x³ + b*x² + c*x + d'
        },
        
        # Экспоненциальные и логарифмические
        'exponential': {
            'func': lambda x, a, b, c: a * np.exp(b * x) + c,
            'n_params': 3,
            'params_names': ['a', 'b', 'c'],
            'p0': [max(y) - min(y), 0.1, min(y)],
            'description': 'a*exp(b*x) + c'
        },
        'logarithmic': {
            'func': lambda x, a, b: a * np.log(x) + b,
            'n_params': 2,
            'params_names': ['a', 'b'],
            'p0': [1.0, 0.0],
            'description': 'a*ln(x) + b'
        },
        'power': {
            'func': lambda x, a, b: a * x**b,
            'n_params': 2,
            'params_names': ['a', 'b'],
            'p0': [1.0, 1.0],
            'description': 'a*x^b'
        },
        
        # Пики (Peaks)
        'gaussian': {
            'func': lambda x, A, x_0, sigma, bg: A * np.exp(-(x - x_0)**2 / (2 * sigma**2)) + bg,
            'n_params': 4,
            'params_names': ['A', 'x_0', 'sigma', 'bg'],
            'p0': [max(y) - min(y), np.mean(x), x_range/6, min(y)],
            'description': 'A*exp(-(x-x_0)²/(2sigma²)) + bg'
        },
        'lorentzian': {
            'func': lambda x, A, x_0, gamma, bg: A * gamma**2 / ((x - x_0)**2 + gamma**2) + bg,
            'n_params': 4,
            'params_names': ['A (высота)', 'x_0 (центр)', 'gamma (ширина)', 'bg (фон)'],
            'p0': [max(y) - min(y), np.mean(x), x_range/10, min(y)],
            'description': 'A*gamma²/((x-x_0)²+gamma²) + bg'
        },
        
        # Тригонометрические
        'sine': {
            'func': lambda x, A, freq, phase, bg: A * np.sin(2 * np.pi * freq * x + phase) + bg,
            'n_params': 4,
            'params_names': ['A', 'частота', 'фаза', 'bg'],
            'p0': [max(y) - min(y), 1.0 / x_range, 0.0, np.mean(y)],
            'description': 'A*sin(2πf*x + φ) + bg'
        },
        'cosine': {
            'func': lambda x, A, freq, phase, bg: A * np.cos(2 * np.pi * freq * x + phase) + bg,
            'n_params': 4,
            'params_names': ['A', 'частота', 'фаза', 'bg'],
            'p0': [max(y) - min(y), 1.0 / x_range, 0.0, np.mean(y)],
            'description': 'A*cos(2πf*x + φ) + bg'
        },
        
        # Двойные пики
        'double_gaussian': {
            'func': lambda x, A1, mu1, sigma1, A2, mu2, sigma2, bg: (
                A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) +
                A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)) + bg
            ),
            'n_params': 7,
            'params_names': ['A1', 'μ1', 'σ1', 'A2', 'μ2', 'σ2', 'bg'],
            'p0': [max(y)/2, x_min + x_range/3, x_range/10, 
                   max(y)/2, x_min + 2*x_range/3, x_range/10, min(y)],
            'description': 'Два гауссовых пика'
        },
        'gaussian_lorentzian': {
            'func': lambda x, A_g, mu_g, sigma, A_l, mu_l, gamma, bg: (
                A_g * np.exp(-(x - mu_g)**2 / (2 * sigma**2)) +
                A_l * gamma**2 / ((x - mu_l)**2 + gamma**2) + bg
            ),
            'n_params': 7,
            'params_names': ['A_g', 'μ_g', 'σ', 'A_l', 'μ_l', 'γ', 'bg'],
            'p0': [max(y)/2, x_min + x_range/3, x_range/10,
                   max(y)/2, x_min + 2*x_range/3, x_range/10, min(y)],
            'description': 'Гаусс + Лоренц'
        }
    }
    
    # Создаем фигуру
    fig = go.FigureWidget()
    
    # Добавляем все точки (серые)
    fig.add_scatter(
        x=x, y=y,
        mode='markers',
        name='All points',
        marker=dict(color='lightgray', size=15, opacity=0.9),
        showlegend=True
    )
    
    # Добавляем выделенные точки (синие)
    fig.add_scatter(
        x=[], y=[],
        mode='markers',
        name='Selected points',
        marker=dict(color='royalblue', size=7, line=dict(color='navy', width=1)),
        showlegend=True
    )
    
    # Добавляем линию фита (красная)
    fig.add_scatter(
        x=[], y=[],
        mode='lines',
        name='Fitting curve',
        line=dict(color='red', width=3),
        showlegend=True
    )
    
    # Добавляем доверительный интервал (опционально)
    fig.add_scatter(
        x=[], y=[],
        mode='lines',
        name='Confidence interval',
        line=dict(color='red', width=1, dash='dash'),
        opacity=0.5,
        showlegend=True
    )
    
    # Настройка графика
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18)
        ),
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_xaxes(
        title='X',
        gridcolor='lightgray',
        gridwidth=1
    )
    fig.update_yaxes(
        title='Y',
        gridcolor='lightgray',
        gridwidth=1
    )
    
    # ============= СОЗДАЕМ ВИДЖЕТЫ =============
    
    # Ползунки для выбора диапазона
    range_slider = widgets.FloatRangeSlider(
        value=[x_min, x_max],
        min=x_min,
        max=x_max,
        step=x_range/200,
        description='X range:',
        continuous_update=True,
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='600px')
    )
    
    # Выбор функции
    func_dropdown = widgets.Dropdown(
        options=[
            ('Linear', 'linear'),
            ('Quadratic', 'quadratic'),
            ('Cubic', 'cubic'),
            ('Exponential', 'exponential'),
            ('Logarithmic', 'logarithmic'),
            ('Power', 'power'),
            ('Gauss', 'gaussian'),
            ('Lorenz', 'lorentzian'),
            ('Sin', 'sine'),
            ('Cos', 'cosine'),
            ('Double Gauss', 'double_gaussian'),
            ('Gauss Lorenz', 'gaussian_lorentzian')
        ],
        value='linear',
        description='Function',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )
    
    # Чекбоксы для дополнительных опций
    show_errors = widgets.Checkbox(
        value=False,
        description='Show parameter errors',
        indent=False,
        layout=widgets.Layout(width='200px')
    )
    
    show_confidence = widgets.Checkbox(
        value=False,
        description='Confidence interval',
        indent=False,
        layout=widgets.Layout(width='200px')
    )
    
    # Кнопки управления
    reset_button = widgets.Button(
        description='Full range',
        button_style='info',
        layout=widgets.Layout(width='150px')
    )
    

    
    # Текст с результатами
    result_text = widgets.HTML(
        value='<b>Fit parameters:</b><br><i>Выберите диапазон...</i>',
        layout=widgets.Layout(
            width='500px',
            height='250px',
            border='1px solid #ccc',
            padding='10px',
            overflow='auto'
        )
    )
    
    # Статистика
    stats_text = widgets.HTML(
        value='<b>Statistics:</b><br><i>Waiting...</i>',
        layout=widgets.Layout(
            width='300px',
            height='250px',
            border='1px solid #ccc',
            padding='10px'
        )
    )
    
    # Информация о точках
    points_info = widgets.HTML(
        value='<b>Number of points:</b> 0',
        layout=widgets.Layout(margin='5px 0px')
    )
    
    # ============= ФУНКЦИИ ДЛЯ ФИТА =============
    
    def fit_selected_data(x_sel, y_sel, func_name):
        """Выполняет фит выбранных данных"""
        if len(x_sel) < 2:
            return None, None, None, None
        
        func_info = functions[func_name]
        
        # Корректируем начальные приближения для некоторых функций
        p0 = func_info['p0'].copy()
        
        # Специальная обработка для разных функций
        if func_name == 'gaussian' or func_name == 'lorentzian':
            # Оцениваем центр пика по максимуму
            peak_idx = np.argmax(y_sel)
            p0[1] = x_sel[peak_idx]  # центр
            p0[0] = y_sel[peak_idx] - p0[3]  # амплитуда
            
        elif func_name == 'exponential' or func_name == 'exponential_decay':
            # Для экспонент нужна положительная скорость
            if len(x_sel) > 2:
                p0[1] = abs(p0[1])
        
        try:
            popt, pcov = optimize.curve_fit(
                func_info['func'],
                x_sel, y_sel,
                p0=p0,
                maxfev=10000
            )
            
            # Стандартные ошибки
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else None
            
            # Вычисляем R²
            y_pred = func_info['func'](x_sel, *popt)
            residuals = y_sel - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_sel - np.mean(y_sel))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # RMSE
            rmse = np.sqrt(np.mean(residuals**2))
            
            return popt, perr, r_squared, rmse
            
        except Exception as e:
            print(f"Fit error: {e}")
            return None, None, None, None
    
    def format_parameters(popt, perr, params_names, show_errors=False):
        """Форматирует параметры для отображения"""
        html = '<table style="width:100%; border-collapse:collapse;">'
        html += '<tr><th style="text-align:left; border-bottom:2px solid #333;">Parameter</th>'
        html += '<th style="text-align:right; border-bottom:2px solid #333;">Value</th>'
        
        if show_errors and perr is not None:
            html += '<th style="text-align:right; border-bottom:2px solid #333;">±</th>'
            html += '<th style="text-align:right; border-bottom:2px solid #333;">Rel error</th>'
        
        html += '</tr>'
        
        for i, (name, val) in enumerate(zip(params_names, popt)):
            html += '<tr>'
            html += f'<td style="text-align:left; padding:2px;">{name}</td>'
            html += f'<td style="text-align:right; padding:2px;">{val:.6f}</td>'
            
            if show_errors and perr is not None:
                rel_error = abs(perr[i] / val * 100) if val != 0 else 0
                html += f'<td style="text-align:right; padding:2px;">±{perr[i]:.6f}</td>'
                html += f'<td style="text-align:right; padding:2px;">({rel_error:.2f}%)</td>'
            
            html += '</tr>'
        
        html += '</table>'
        return html
    
    # ============= ФУНКЦИЯ ОБНОВЛЕНИЯ =============
    
    def update_plot(change=None):
        """Обновляет график при изменении параметров"""
        
        # Получаем текущий диапазон
        x_min_curr, x_max_curr = range_slider.value
        
        # Выбираем точки в диапазоне
        mask = (x >= x_min_curr) & (x <= x_max_curr)
        x_sel = x[mask]
        y_sel = y[mask]
        
        n_points = len(x_sel)
        points_info.value = f'<b>Number of points:</b> {n_points}'
        
        # Обновляем выделенные точки
        fig.data[1].x = x_sel
        fig.data[1].y = y_sel
        
        # Минимальное количество точек для фита
        min_points = functions[func_dropdown.value]['n_params']
        
        if n_points >= min_points:
            # Выполняем фит
            popt, perr, r_squared, rmse = fit_selected_data(
                x_sel, y_sel, 
                func_dropdown.value
            )
            
            if popt is not None:
                func_info = functions[func_dropdown.value]
                
                # Создаем плавную линию для фита
                x_fit = np.linspace(x_min_curr, x_max_curr, 200)
                y_fit = func_info['func'](x_fit, *popt)
                
                # Обновляем линию фита
                fig.data[2].x = x_fit
                fig.data[2].y = y_fit
                
                # Доверительный интервал
                if show_confidence.value and perr is not None:
                    # Простая оценка доверительного интервала
                    y_fit_upper = y_fit + 2 * rmse
                    y_fit_lower = y_fit - 2 * rmse
                    
                    fig.data[3].x = np.concatenate([x_fit, x_fit[::-1]])
                    fig.data[3].y = np.concatenate([y_fit_upper, y_fit_lower[::-1]])
                    fig.data[3].fill = 'toself'
                    fig.data[3].opacity = 0.2
                else:
                    fig.data[3].x = []
                    fig.data[3].y = []
                
                # Форматируем результаты
                params_html = '<b>Fit parameters:</b><br>'
                params_html += format_parameters(
                    popt, perr, 
                    func_info['params_names'],
                    show_errors.value
                )
                
                stats_html = '<b>Statistics:</b><br>'
                stats_html += f'<b>R²:</b> {r_squared:.6f}<br>'
                stats_html += f'<b>RMSE:</b> {rmse:.6f}<br>'
                stats_html += f'<b>Number of points:</b> {n_points}<br>'
                stats_html += f'<b>Number of parameters:</b> {len(popt)}<br>'
                stats_html += f'<b>Function</b> {func_info["description"]}'
                
                result_text.value = params_html
                stats_text.value = stats_html
                
            else:
                # Очищаем при ошибке
                fig.data[2].x = []
                fig.data[2].y = []
                fig.data[3].x = []
                fig.data[3].y = []
                
                result_text.value = '<b>Fitting error</b><br><i>Try another range or another function</i>'
                stats_text.value = '<b>Statistics:</b><br>Calculation error'
        else:
            # Очищаем при недостатке точек
            fig.data[2].x = []
            fig.data[2].y = []
            fig.data[3].x = []
            fig.data[3].y = []
            
            result_text.value = f'<b>Not enough points</b><br>Min threshold: {min_points}, current number: {n_points}'
            stats_text.value = f'<b>Statistics:</b><br>Number of points: {n_points}/{min_points}'
    
    # ============= ОБРАБОТЧИКИ =============
    
    def on_range_change(change):
        """Обработчик изменения диапазона"""
        update_plot()
    
    def on_function_change(change):
        """Обработчик смены функции"""
        update_plot()
    
    def on_options_change(change):
        """Обработчик изменения опций"""
        update_plot()
    
    def reset_range(b):
        """Сброс диапазона на весь массив"""
        range_slider.value = [x_min, x_max]
    
    def manual_fit(b):
        """Принудительное обновление фита"""
        update_plot()
    
    # Подключаем обработчики
    range_slider.observe(on_range_change, 'value')
    func_dropdown.observe(on_function_change, 'value')
    show_errors.observe(on_options_change, 'value')
    show_confidence.observe(on_options_change, 'value')
    reset_button.on_click(reset_range)
    #fit_button.on_click(manual_fit)
    
    # Первое обновление
    update_plot()
    
    # ============= СБОРКА ИНТЕРФЕЙСА =============
    
    # Верхняя панель с кнопками
    top_panel = widgets.HBox([
        func_dropdown,
        widgets.VBox([
            show_errors,
            show_confidence
        ]),
        reset_button,
        #fit_button
    ], layout=widgets.Layout(justify_content='flex-start', margin='5px 0px'))
    
    # Панель с результатами
    results_panel = widgets.HBox([
        result_text,
        stats_text
    ], layout=widgets.Layout(margin='10px 0px'))
    
    # Собираем все вместе
    ui = widgets.VBox([
        widgets.HTML(f'<h2 style="color:#2c3e50;">{title}</h2>'),
        widgets.HTML('<hr style="margin:5px 0px;">'),
        top_panel,
        range_slider,
        points_info,
        results_panel,
        #widgets.HTML('<i style="color:#7f8c8d;">Move the sliders</i>')
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    return ui, fig

