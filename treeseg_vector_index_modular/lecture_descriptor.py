from dataclasses import dataclass


@dataclass(frozen=True)
class LectureDescriptor:
    speaker: str
    course_dir: str
    meeting_id: str
    video_id: str
    transcripts_path: str
    meeting_dir: str

    @property
    def key(self):
        return f"{self.speaker}/{self.course_dir}/{self.meeting_id}"

    @property
    def label(self):
        return f"{self.key} (video_id={self.video_id})"
