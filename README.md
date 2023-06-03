# YoshAI

An AI to play Mario Kart for you. 

## Overview

YoshAI is a project that aims to create an AI that can play Mario Kart Wii at least as
effectively as a human. It uses the [Dolphin emulator](https://dolphin-emu.org/) as
a simulation platform. At a basic level, YoshAI starts dolphin, finds the window
displaying the game, and uses the pixels of the window to predict the best controls
to send to dolphin. In collect mode, a user uses a gamepad to control the character
and YoshAI saves the images of the game and user inputs as data to be used for training.
Train mode uses behavior cloning to train a convolutional neural network to copy the
user's inputs. Run mode uses the trained model to play Mario Kart. Dagger mode [TODO]

## Setup

Note: this project was developed and tested on Ubuntu 22.04. Most parts of the project
will likely not work on other operating systems.

Note 2: I used an Xbox 360 controller for this project. Some pieces of this project
are specific to an Xbox 360 controller, but would be easy to modify for another controller.

### Python Dependencies

Use `pip install -r requirements.txt` to install all necessary python packages:
```
xlib            used to capture pixel data from dolphin
inputs          used to capture input data from controller
torch           used for training
torchvision     used for training
av              used by torchvision to save videos of runs
python-uinput   used to simulate controller inputs to doplhin
numpy           used for general computation, will likely be removed soon
```

Additional setup is required to make the `python-uinput` module work correctly.

### `uinput`
(Required for emulating controller output)

The `uinput` module (used to emulate a controller or other input device)
requires certain permissions to use. By default, the device `/dev/uinput`
belongs to the root user, so the easiest way to obtain these permissions
is to run the project as root. You probably shouldn't do that unless you 
really trust me, so alternatively, you can create a group to assign the
device to. Run the following commands to set up the necessary permissions
(you may need to run as root by first running `sudo -i`):

```bash
sudo groupadd --system uinput
sudo echo 'SUBSYSTEM=="misc", KERNEL=="uinput", GROUP="uinput", MODE="0660"' >> /etc/udev/rules.d/99-uinput.rules
sudo udevadm trigger /dev/uinput
sudo modprobe uinput
ls -l /dev/uinput
```

You should see the uinput group listed in the output of the last command:

```
crw-rw---- 1 root uinput 10, 223 May 27 15:46 /dev/uinput
```

Next add your user to the `uinput` group (run as your own user, not as root):

```bash
sudo usermod -a -G uinput $USER
```

Finally, log out of your account and log back in and check that you were
added to the group by running the command `groups` and checking for `uinput` in the result.
You can verify that the full setup was successful by running 
`python virtual_controller.py`. If you don't see any error messages, you're good to go!

### Dolphin

You'll need to have a recent version of Dolphin emulator installed on your computer.
To get the latest version of Dolphin, you'll need to build from source. Follow these
instructions to build and install dolphin: 
[Building Dolphin](https://wiki.dolphin-emu.org/index.php?title=Building_Dolphin_on_Linux)

Note that though the link encourages using a ppa to install dolphin, on most distributions
of Linux, the version of dolphin from any ppa is not current enough to work with this project.
I had better luck using the latest nightly build from the Dolphin GitHub repo rather than
checking out a release tag as the instructions directed.

Once dolphin is installed, there's a couple more steps to get it configured correctly.

First, you'll need to get a copy of the iso file for Mario Kart Wii. Since this is the
intellectual property of Nintendo, I have not distributed this file as part of this repo.
Legally, you should buy a disc copy of Mario Kart Wii, then rip the file from the disc to
use with Dolphin. That's a bit pricey, though... Whatever you do, you should definitely not
just Google "Mario Kart Wii iso", because you would likely find the file for free online
instead of paying for it through the proper channels, and Nintendo might lose out on $50
of profit. Once you have the iso file, place it in the root of this project directory and
update the `GAME_FILE` variable in `dolphin.py` if necessary. You should also place a second
copy of this file in dolphin's game folder (location is configurable in dolphin).

Next, open up dolphin and navigate to `Options` > `Graphics Settings`. In the `General`
tab, `Other` section, check the box labeled "Auto-Adjust Window Size". This will keep
the size of the window consistent across restarts of the game.

Lastly, you'll need to do some controller configuration:

1. Copy the wiimote profiles from the `wiimote_profiles` directory to the dolphin `Profiles`
   directory. For me, this is located at `~/.config/dolphin-emu/Profiles`. If this directory
   does not exist, the easiest way to make sure you're in the right place is to create a wiimote
   profile in the dolphin UI, name it something identifiable, save it, and search your file system
   for the profile you created.
   (See [this link](https://dolphin-emu.org/docs/guides/configuring-controllers/) for instructions).
2. Connect your controller and modify the `mario_kart_linux` profile to work with your controller.
   You will likely need to select the input device make sure that all the necessary controls are
   working. Make sure to **save the profile** when you're done. If you modify the controls
   significantly, you may need to make changes to `controller.py` to ensure that the correct inputs
   are captured.
3. Load the `mario_kart_linux_uinput` profile and then run `python virtual_controller.py`. While
   this file is running, refresh the available inputs and make sure that the selected one for this
   profile is connected. Also make sure that your physical controller is not selected (this profile
   is used for input simulation). If needed, select a different controller from the list and **save
   the profile**.
4. Go back to the main dolphin window, right click on Mario Kart Wii, and go to `Properties`.
   Navigate to `Game Config` and `Editor`, then in the `User Config` section, click on `Presets` >
   `Editor` > `Open in External Editor`. A file will open in your text editor. Use the text editor
   to find the location of this file, and make sure that `GAME_INI_FILE` in `dolphin.py` is set to
   the location of this file. This file will be overwritten to allow YoshAI to programmatically
   change game settings, so if you have any of your own settings in this file, back them up, and
   if desired, make some simple changes to `dolphin.py` to avoid deleting your data.

#### Save files

Included in this repo are a few save files that are used to skip programmatically navigating
through the UI of Mario Kart. Each save file brings the game to the point after selecting a
character, vehicle, and course. You can create your own save files by starting Mario Kart,
navigating through each selection process until you reach the "Go" screen, then in the dolphin
main window, go to `Emulation` > `Save State` > `Save file` and enter a file name. Place any
saves that you want to be accessible to YoshAI in the `saves/` folder in this directory.

All development of this project was done with the simplest game mode in mind: solo time trials.
When creating your own save states, I recommend keeping with the simple theme. To reach the
simplest possible game, make sure to select "Solo Time Trials" after picking a course instead
of racing a ghost. That being said, nothing in this project precludes you from using any other
single player game mode in Mario Kart. If you're feeling ambitious, you could create save states
for races against AIs, or even battles. Any save state that immediately loads to the "Go"
button is a valid state for YoshAI.

## Usage

Each mode follows the basic format of:

```bash
python yoshai.py [mode] [arguments]
```

Available modes are `collect`, `train`, `run`, `dagger`, and `reformat`. 
More details on each mode are below.

### `collect`

Basic usage:

```bash
python yoshai.py collect
```

Optional arguments:

```bash
python yoshai.py collect -c [course]
```

`course` should be the name of a save file in the `saves/` directory in this folder. Do not
include the `.sav` extension in the argument. Default value for `course` is `luigi-circuit`.

Collected data will be saved in the `bcdata/` directory, named using the save file name and
a unique index.

Make sure your controller is connected before starting collect mode. Dolphin will be started
using the provided save file, and once YoshAI is ready to record data, a message will be
printed to the terminal. After the message appears, make sure the game window is in focus,
then press A. YoshAI will start recording your inputs and Dolphin will start the game. Once
the course is finished, go to the terminal and press Ctrl-C to stop the recording. YoshAI will
then save your final data to a file for later training.

### `train`

Basic usage:

```bash
python yoshai.py train -d bcdata
```

Optional arguments:

```bash
python yoshai.py train -d [data_path] -b [batch_size] -l [learning_rate] --decay [decay] -e [epochs] -m [model_path] -g [gamma]

data_path: folder (or file) containing data to train with
batch_size: number of datapoints to train with at once - default: 64
learning_rate: optimizer initial learning rate, may change if gamma is not 0 - default: 0.01
decay: optimizer weight decay - default: 0
epochs: number of epochs to train for - default: 10
model_path: if provided, best model is saved to this path
gamma: exponential factor to lower learning rate each epoch, should be less than 1 - default: 0.9
```

Train will run for the given number of epochs and then stop. After each epoch, if the model improved,
the current model will be saved to disk, meaning you can stop training at any time by Ctrl-C and still
keep your best results so far.

### `run`

Basic usage:

```bash
python yoshai.py run -m model.pt
```

Optional arguments:

```bash
python yoshai.py run -m [model_path] -c [course] --save_video

model_path: saved model to use for predicting controls
course: save file of course to run, must be in saves/, do not include .sav
save_video: if present, video of the run will be saved after the run is stopped
```

Similar to `collect`, this mode will start dolphin, but will also start a virtual controller.
Controller inputs will be predicted by the given model. The course will be automatically started,
but must be stopped manually (by Ctrl-C) when it's done (or earlier). After stopping, the video
will be saved if the `save_video` argument was provided.

### `dagger`

Not yet implemented.

### `reformat`

This mode was used during development to change the format of saved data.
You should not depend on its behavior.

## Acknowledgements

[TODO]

## Next Steps

- remove numpy dependency in `mario_dataset.py`
- make controller config more robust
- add interactive prompts to collect system
- make data storage paths easier to change with arguments
- make generated file names robust to duplicates