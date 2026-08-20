if __name__ == "__main__":
    try:
        import fire
        from src.command_line_interface import CommandLineInterface

        fire.Fire(CommandLineInterface)
    except KeyboardInterrupt:
        print("Keyboard Interrupted...")
