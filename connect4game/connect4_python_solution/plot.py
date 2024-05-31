import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8]
y1 = [1, 0.6298, 0.4855, 0.4120, 0.3123, 0.3786, 0.4347, 0.3829]
y2 = [1, 0.6785, 0.6283, 0.4983, 0.4306, 0.4304, 0.3987, 0.3709]
y3 = [1, 0.5409, 0.5877, 0.4986, 0.3963, 0.4245, 0.3814, 0.3531]

plt.figure(figsize=(10, 6))

plt.plot(x, y1, marker='o', label='level1')
plt.plot(x, y2, marker='o', label='level2')
plt.plot(x, y3, marker='o', label='level3')

plt.xlabel('Broj procesora')
plt.ylabel('Učinkovitost')

plt.xticks(range(1, 9))
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])

plt.legend()

plt.grid(True)
plt.show()
