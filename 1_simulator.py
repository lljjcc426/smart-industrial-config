import pygame
import sys
import random

# 配置
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
COLOR_BG = (10, 15, 20)
COLOR_GRID = (30, 40, 50)
COLOR_ANCHOR_BORDER = (0, 200, 255)
COLOR_TEXT_GLOW = (0, 255, 0)
COLOR_ALERT = (255, 50, 50)

def init_sim():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Industrial SCADA (Single LCD v3.0)")
    
    fonts = ["consolas", "arial", "microsoftyahei"]
    font_name = pygame.font.match_font(fonts[0]) or pygame.font.get_default_font()
    
    font_anchor = pygame.font.Font(font_name, 26)
    font_val = pygame.font.Font(font_name, 100)
    font_btn = pygame.font.Font(font_name, 22)
        
    return screen, font_anchor, font_val, font_btn

class ScadaApp:
    def __init__(self):
        self.screen, self.f_anchor, self.f_val, self.f_btn = init_sim()
        self.clock = pygame.time.Clock()
        self.pressure = 60.0
        self.venting = False
        self.btn_vent_rect = pygame.Rect(WINDOW_WIDTH//2 - 110, 450, 220, 60)
        self.flicker = False

    def update_physics(self):
        if not self.venting:
            self.pressure += random.uniform(0.1, 0.5)
            self.pressure = min(150.0, self.pressure)
        else:
            self.pressure -= random.uniform(2.0, 4.0)
            if self.pressure <= 50: 
                self.venting = False

        self.flicker = random.random() < 0.02
        return int(self.pressure)

    def draw_lcd(self, center_x, center_y, value):
        lcd_rect = pygame.Rect(center_x - 120, center_y - 60, 240, 120)
        alert = value > 120
        color = COLOR_ALERT if alert else COLOR_TEXT_GLOW
        if self.flicker and alert:
            color = (random.randint(200,255), random.randint(0,50), random.randint(0,50))
        
        pygame.draw.rect(self.screen, (0, 0, 0), lcd_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), lcd_rect, 2)
        txt = self.f_val.render(f"{value:03d}", True, color)
        self.screen.blit(txt, txt.get_rect(center=lcd_rect.center))

    def draw(self):
        self.screen.fill(COLOR_BG)
        # 网格
        for x in range(0, WINDOW_WIDTH, 40):
            pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, WINDOW_HEIGHT), 1)
        for y in range(0, WINDOW_HEIGHT, 40):
            pygame.draw.line(self.screen, COLOR_GRID, (0, y), (WINDOW_WIDTH, y), 1)

        val = self.update_physics()

        # 锚点文字
        pygame.draw.rect(self.screen, (0, 40, 40), (WINDOW_WIDTH//2 - 225, 60, 450, 50))
        pygame.draw.rect(self.screen, COLOR_ANCHOR_BORDER, (WINDOW_WIDTH//2 - 225, 60, 450, 50), 3)
        txt_anchor = self.f_anchor.render("REACTOR A - PRESSURE MONITOR", True, (200, 255, 255))
        self.screen.blit(txt_anchor, (WINDOW_WIDTH//2 - 205, 72))

        # 单LCD居中
        self.draw_lcd(WINDOW_WIDTH//2, 220, val)

        # 按钮居中
        btn_color = (50, 50, 200) if self.btn_vent_rect.collidepoint(pygame.mouse.get_pos()) else (20, 20, 150)
        pygame.draw.rect(self.screen, btn_color, self.btn_vent_rect, border_radius=5)
        txt_btn = self.f_btn.render("EMERGENCY VENT", True, (255, 255, 255))
        self.screen.blit(txt_btn, txt_btn.get_rect(center=self.btn_vent_rect.center))

        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.btn_vent_rect.collidepoint(event.pos):
                        print(">>> Manual Venting Triggered!")
                        self.venting = True
            self.draw()
            self.clock.tick(30)

if __name__ == "__main__":
    ScadaApp().run()