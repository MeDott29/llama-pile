import os
from pathlib import Path
import subprocess
import json
from datetime import datetime, timedelta
import random
from typing import List, Dict, Optional
import shutil
from dataclasses import dataclass
import ffmpeg
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

@dataclass
class VideoMetadata:
    filename: str
    size_mb: float
    duration: str
    resolution: str
    codec: str
    audio_tracks: int
    subtitle_tracks: int
    creation_date: str
    last_modified: str

class SRTGenerator:
    def __init__(self):
        self.phrases = [
            "This is an interesting scene",
            "The plot thickens",
            "Something important happens here",
            "A crucial moment",
            "The story develops",
            "Watch carefully now",
            "Key details emerge",
            "Remember this part",
            "Significant development",
            "Notable event occurs"
        ]
    
    def generate_timestamp(self, start: timedelta) -> tuple[str, str]:
        """Generate SRT timestamp format"""
        end = start + timedelta(seconds=random.uniform(2, 5))
        return (
            f"{str(start).split('.')[0]},000",
            f"{str(end).split('.')[0]},000"
        )
    
    def create_srt(self, duration: float, output_path: Path):
        """Create random SRT file with reasonable timing"""
        current_time = timedelta()
        end_time = timedelta(seconds=float(duration))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            counter = 1
            while current_time < end_time:
                start_stamp, end_stamp = self.generate_timestamp(current_time)
                f.write(f"{counter}\n")
                f.write(f"{start_stamp} --> {end_stamp}\n")
                f.write(f"{random.choice(self.phrases)}\n\n")
                
                current_time += timedelta(seconds=random.uniform(5, 10))
                counter += 1

class MKVManager:
    def __init__(self, videos_dir: str = "/home/ath/Videos"):
        self.videos_dir = Path(videos_dir).expanduser()  # Expand user path
        self.output_dir = self.videos_dir / "processed"
        self.console = Console()
        self.srt_generator = SRTGenerator()
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed
    
    def scan_files(self) -> List[Path]:
        """Scan directory for MKV files"""
        return list(self.videos_dir.glob("**/*.mkv"))
    
    def get_metadata(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from MKV file using ffmpeg"""
        try:
            probe = ffmpeg.probe(str(video_path))
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            # Count audio and subtitle tracks
            audio_tracks = len([s for s in probe['streams'] if s['codec_type'] == 'audio'])
            subtitle_tracks = len([s for s in probe['streams'] if s['codec_type'] == 'subtitle'])
            
            return VideoMetadata(
                filename=video_path.name,
                size_mb=video_path.stat().st_size / (1024 * 1024),
                duration=probe['format'].get('duration', 'unknown'),
                resolution=f"{video_info.get('width', '?')}x{video_info.get('height', '?')}",
                codec=video_info.get('codec_name', 'unknown'),
                audio_tracks=audio_tracks,
                subtitle_tracks=subtitle_tracks,
                creation_date=datetime.fromtimestamp(video_path.stat().st_ctime).strftime('%Y-%m-%d'),
                last_modified=datetime.fromtimestamp(video_path.stat().st_mtime).strftime('%Y-%m-%d')
            )
        except Exception as e:
            self.console.print(f"[red]Error reading metadata for {video_path.name}: {str(e)}")
            return None

    def display_files(self, files: List[Path]):
        """Display video files in a formatted table"""
        table = Table(title="MKV Files Overview")
        
        table.add_column("Filename", style="cyan")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Duration")
        table.add_column("Resolution")
        table.add_column("Codec")
        table.add_column("Audio Tracks", justify="right")
        table.add_column("Subtitles", justify="right")
        table.add_column("Modified Date")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Reading metadata...", total=len(files))
            
            for file in files:
                metadata = self.get_metadata(file)
                if metadata:
                    table.add_row(
                        metadata.filename,
                        f"{metadata.size_mb:.1f}",
                        metadata.duration,
                        metadata.resolution,
                        metadata.codec,
                        str(metadata.audio_tracks),
                        str(metadata.subtitle_tracks),
                        metadata.last_modified
                    )
                progress.advance(task)
        
        self.console.print(table)

    def transform_video(self, input_path: Path, output_format: str = 'mp4', 
                       quality: str = 'high', compress: bool = False):
        """Transform MKV to different format with Twitter-compatible settings"""
        output_path = self.output_dir / f"{input_path.stem}.{output_format}"
        srt_path = self.output_dir / f"{input_path.stem}.srt"
        
        # Twitter video requirements:
        # - Maximum resolution: 1920x1200
        # - Maximum frame rate: 40fps
        # - Maximum duration: 2 minutes and 20 seconds
        # - Maximum file size: 512MB
        # - Recommended codec: H.264 with AAC audio
        try:
            probe = ffmpeg.probe(str(input_path))
            duration = float(probe['format']['duration'])
            
            # Generate SRT file
            self.srt_generator.create_srt(duration, srt_path)
            
            stream = ffmpeg.input(str(input_path))
            
            # Twitter-optimized settings
            stream = ffmpeg.output(
                stream, 
                str(output_path),
                **{
                    'c:v': 'libx264',  # H.264 codec
                    'b:v': '5M',       # 5Mbps video bitrate
                    'maxrate': '5M',
                    'bufsize': '10M',
                    'vf': 'scale=1280:-2',  # Scale to 720p while maintaining aspect ratio
                    'r': 30,          # 30fps
                    'c:a': 'aac',     # AAC audio codec
                    'b:a': '128k',    # 128kbps audio bitrate
                    'ar': 44100,      # 44.1kHz audio sample rate
                    'preset': 'medium',
                    'pix_fmt': 'yuv420p'  # Required pixel format for maximum compatibility
                }
            ).overwrite_output()
            
            # Run the conversion
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Converting {input_path.name}...")
                ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
                progress.update(task, completed=True)
            
            self.console.print(f"[green]Successfully converted {input_path.name} to {output_path.name}")
            self.console.print(f"[green]Generated SRT file: {srt_path.name}")
            return output_path
            
        except ffmpeg.Error as e:
            self.console.print(f"[red]Error converting {input_path.name}: {str(e)}")
            return None

    def batch_process(self, files: List[Path], output_format: str = 'mp4', 
                     quality: str = 'high', compress: bool = False):
        """Process multiple video files with progress tracking"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Processing videos...", total=len(files))
            
            for file in files:
                progress.update(task, description=f"Processing {file.name}")
                self.transform_video(file, output_format, quality, compress)
                progress.advance(task)

    def interactive_menu(self):
        """Display interactive menu for file operations"""
        while True:
            self.console.clear()
            self.console.print("\n[cyan]MKV File Manager[/cyan]")
            self.console.print("\n1. List all MKV files")
            self.console.print("2. Convert selected files")
            self.console.print("3. Batch convert all files")
            self.console.print("4. Compress videos")
            self.console.print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == "1":
                files = self.scan_files()
                self.display_files(files)
                input("\nPress Enter to continue...")
                
            elif choice == "2":
                files = self.scan_files()
                self.display_files(files)
                indices = input("\nEnter file numbers to convert (comma-separated): ").split(",")
                try:
                    selected_files = [files[int(i.strip()) - 1] for i in indices]
                    format_choice = input("Enter output format (mp4/mkv): ").lower()
                    quality = input("Enter quality (high/medium/low): ").lower()
                    self.batch_process(selected_files, format_choice, quality)
                except (ValueError, IndexError) as e:
                    self.console.print(f"[red]Invalid selection: {str(e)}")
                input("\nPress Enter to continue...")
                
            elif choice == "3":
                files = self.scan_files()
                format_choice = input("Enter output format (mp4/mkv): ").lower()
                quality = input("Enter quality (high/medium/low): ").lower()
                self.batch_process(files, format_choice, quality)
                input("\nPress Enter to continue...")
                
            elif choice == "4":
                files = self.scan_files()
                self.display_files(files)
                indices = input("\nEnter file numbers to compress (comma-separated): ").split(",")
                try:
                    selected_files = [files[int(i.strip()) - 1] for i in indices]
                    self.batch_process(selected_files, compress=True)
                except (ValueError, IndexError) as e:
                    self.console.print(f"[red]Invalid selection: {str(e)}")
                input("\nPress Enter to continue...")
                
            elif choice == "5":
                self.console.print("[green]Goodbye!")
                break

    def get_latest_mkv(self) -> Optional[Path]:
        """Get the most recently created MKV file"""
        mkv_files = list(self.videos_dir.glob("*.mkv"))
        if not mkv_files:
            self.console.print("[yellow]No MKV files found in directory")
            return None
        
        return max(mkv_files, key=lambda x: x.stat().st_ctime)

def main():
    manager = MKVManager()  # No need to ask for directory path
    latest_file = manager.get_latest_mkv()
    
    if latest_file:
        manager.console.print(f"[green]Processing latest MKV file: {latest_file.name}")
        manager.transform_video(latest_file)
    else:
        manager.console.print("[red]No MKV files found to process")

if __name__ == "__main__":
    main()
