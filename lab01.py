import matplotlib.pyplot as plt
import numpy as np

# Our data: 2D points coordinates (x, y)
x, y = [], []
title = 'Click to add Points.\nPress Enter for Linear Regression.\nPress Escape to Reset.'
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)
ax.set_title(title)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Mouse click event handler
def on_click(event):
    if event.inaxes is None:
        return

    # Add x and y coordinates of the point to the list
    x.append(event.xdata)
    y.append(event.ydata)

    # Draw the point
    ax.plot(event.xdata, event.ydata, 'ro')

    fig.canvas.draw_idle()

# Keyboard press event handler
def on_key(event):
    if event.key == 'enter': # Enter key is pressed
        # Perform Linear Regression in Closed Form
        w, b = linear_regression(x, y)
        # Draw the line from two end-points (x0, y0)
        # and (x1, y1)
        x0, y0 = 0, b
        x1, y1 = 10, w * 10 + b
        ax.plot([x0, x1], [y0, y1], 'b-')
    if event.key == 'escape': # ESC key is pressed
        ax.clear()
        ax.set_title(title)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    fig.canvas.draw_idle()

def linear_regression(x, y):
    # Complete this function by implementing the
    # Closed-form equations we saw in the lectures
    x = np.array(x)
    y = np.array(y)
    A = [[np.sum(x*x), np.sum(x)],
         [np.sum(x), len(x)]]
    A = np.array(A)
    c = [[np.sum(x*y)],
         [np.sum(y)]]
    c = np.array(c)
    w, b = np.linalg.inv(A) @ c
    return w, b

fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
