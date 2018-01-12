import pylab as plt
import numpy as np
import matplotlib.animation as animation
fig = plt.figure(figsize=(10,5))
ims = []
l = 1.0
obs = env.reset()
R,t,done = 0, 0, False
while not done and t < 200:
    action = agent.act(obs)
    print("t:{} obs:{} action:{}".format(t,obs,action))
    im = plt.plot([-2,2],[0,0],color="black")
    im = plt.plot([obs[1],obs[1]+l*np.sin(obs[3])],[0,l*np.cos(obs[3])],
                  "o-",color="blue",lw=4,label="Pole")
    ims.append(im)
    obs, r, done, _ = env.step(action)
    R += r
    t += 1
#     print("test episode : {} R: {}".format(i,R))
agent.stop_episode()
plt.legend()
plt.xlim(-2.0,2.0)
plt.ylim(-1.0,1.0)
ani = animation.ArtistAnimation(fig, ims, interval=100)
ani.save("animation.gif", writer="imagemagick")