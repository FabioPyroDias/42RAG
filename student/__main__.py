import fire
from student.command_line_interface import CommandLineInterface

if __name__ == "__main__":
    try:
        fire.Fire(CommandLineInterface)
    except KeyboardInterrupt:
        print("Keyboard Interrupted...")
