from pathlib import Path

from .lecture_descriptor import LectureDescriptor


class LectureCatalog:
    @staticmethod
    def discover_lectures(data_dir, speaker=None, course_dir=None, meeting_ids=None):
        base = Path(data_dir)
        if speaker:
            speakers = [speaker]
        else:
            speakers = [d.name for d in base.iterdir() if d.is_dir()]

        if meeting_ids is None:
            meeting_ids = []
        elif isinstance(meeting_ids, str):
            meeting_ids = [meeting_ids]

        lectures = []
        for sp in sorted(speakers):
            sp_dir = base / sp
            if not sp_dir.is_dir():
                continue
            if course_dir:
                courses = [course_dir]
            else:
                courses = [d.name for d in sp_dir.iterdir() if d.is_dir()]
            for course in sorted(courses):
                course_path = sp_dir / course
                if not course_path.is_dir():
                    continue
                if meeting_ids:
                    meetings = meeting_ids
                else:
                    meetings = [d.name for d in course_path.iterdir() if d.is_dir()]
                for meeting_id in sorted(meetings):
                    meeting_dir = course_path / meeting_id
                    if not meeting_dir.is_dir():
                        continue
                    transcripts_path = None
                    for fn in meeting_dir.iterdir():
                        if fn.name.endswith("_transcripts.csv"):
                            transcripts_path = fn
                            break
                    if transcripts_path is None:
                        continue
                    video_id = transcripts_path.name.replace("_transcripts.csv", "")
                    lectures.append(
                        LectureDescriptor(
                            speaker=sp,
                            course_dir=course,
                            meeting_id=meeting_id,
                            video_id=video_id,
                            transcripts_path=str(transcripts_path),
                            meeting_dir=str(meeting_dir),
                        )
                    )
        return lectures

    @staticmethod
    def format_lecture_list(lectures):
        lines = []
        for idx, lecture in enumerate(lectures, start=1):
            lines.append(f"{idx:02d}. {lecture.label}")
        return "\n".join(lines)

    @staticmethod
    def resolve_lecture_choice(lectures, choice, allow_all=True):
        if not choice:
            return None
        value = choice.strip()
        if not value:
            return None
        lowered = value.lower()
        if lowered in {"all", "a"}:
            if allow_all:
                return None
            raise ValueError("Global search is disabled. Pick a lecture.")
        if value.isdigit():
            idx = int(value) - 1
            if 0 <= idx < len(lectures):
                return lectures[idx].key
            raise ValueError("Lecture index out of range.")
        exact = [lec for lec in lectures if lec.key == value]
        if exact:
            return exact[0].key
        meeting_matches = [lec for lec in lectures if lec.meeting_id == value]
        if len(meeting_matches) == 1:
            return meeting_matches[0].key
        if meeting_matches:
            raise ValueError("Meeting id matches multiple lectures. Use full key.")
        raise ValueError("Unknown lecture choice.")
