import math
import turtle

# Setup screen
screen = turtle.Screen()
screen.bgcolor("black")   # Black background
screen.title("Proposal Heart")

# Setup turtle
pen = turtle.Turtle()
pen.speed(0)
pen.color("red")
pen.hideturtle()

# Heart curve functions
def hearta(k):
    return 15 * math.sin(k) ** 3

def heartb(k):
    return 12 * math.cos(k) - 5*math.cos(2*k) - 2*math.cos(3*k) - math.cos(4*k)

# Draw heart outline + spider web lines
for i in range(0, 360):
    k = math.radians(i)
    x = hearta(k) * 20
    y = heartb(k) * 20
    
    # Go to point on heart
    pen.goto(x, y)
    
    # Draw "web" line back to center
    pen.goto(0, 0)
    
    # Return to the heart point (so outline connects too)
    pen.goto(x, y)

# Function to write glowing text
def write_glow(text, x, y):
    glow_pen = turtle.Turtle()
    glow_pen.hideturtle()
    glow_pen.penup()
    glow_pen.goto(x, y)
    font_style = ("Comic Sans MS", 26, "bold italic")
    
    # Glow effect: draw text around the main position
    glow_pen.color("red")
    for dx, dy in [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2),(-2,2),(2,-2)]:
        glow_pen.goto(x+dx, y+dy)
        glow_pen.write(text, align="center", font=font_style)
    
    # Main white text on top
    glow_pen.goto(x, y)
    glow_pen.color("white")
    glow_pen.write(text, align="center", font=font_style)

# Write glowing proposal inside heart
write_glow("juju will you marry me?", 0, -20)

# Keep window open
turtle.done()
