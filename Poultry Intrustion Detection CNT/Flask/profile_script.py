import cProfile
import pstats
import io
from app import app  # Adjust import based on your file structure

def profile_flask_app():
    app.run(debug=True)

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    profile_flask_app()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
    ps.print_stats()
    print(s.getvalue())
