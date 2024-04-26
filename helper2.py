import matplotlib.pyplot as plt
from IPython import display
import numpy as np

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize = (20,6))
def plot(times, mean_time,rewards,mean_rewards,loss, mean_loss):
     
     ax1.clear() 
     fig.suptitle('Graphing')
     ax1.set_title('Time to completion')
     ax1.set_xlabel('Number of Games')
     ax1.set_ylabel('Time')
     ax1.plot(times, label = 'Times')
     ax1.plot(mean_time, label = 'Mean Times')
     ax1.set_ylim(ymin=0)
     ax1.text(len(times)-1, times[-1], str(times[-1]))
     ax1.text(len(mean_time)-1, mean_time[-1], str(mean_time[-1]))

     
     ax2.clear()
     ax2.set_title('Rewards')
     ax2.set_xlabel('Number of Games')
     ax2.set_ylabel('Rewards')
     ax2.plot(rewards)
     ax2.plot(mean_rewards)
     ax2.set_ylim(ymin=-10)
     ax2.text(len(rewards)-1, rewards[-1], str(rewards[-1]))
     ax2.text(len(mean_rewards)-1, mean_rewards[-1], str(mean_rewards[-1]))

     ax3.clear()
     ax3.set_title('Loss')
     ax3.set_xlabel('Number of Games')
     ax3.set_ylabel('Loss')
     ax3.plot(loss)
     ax3.plot(mean_loss)
     ax3.set_ylim(ymin=0)
     ax3.text(len(loss)-1, loss[-1], str(loss[-1]))
     ax3.text(len(mean_loss)-1, mean_loss[-1], str(mean_loss[-1]))
     
     display.display(fig)
     display.clear_output(wait=True)
     plt.pause(0.1)

